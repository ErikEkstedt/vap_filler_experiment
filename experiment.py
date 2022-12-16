import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from os.path import basename, exists, join
from glob import glob
from tqdm import tqdm
import parselmouth
import json
import matplotlib.pyplot as plt
import numpy as np

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

from vap.model import VAPModel
from vap.utils import read_txt, read_json
from vap.audio import load_waveform, log_mel_spectrogram


"""
1. Extract all filler utterances
    - is_filler: 'uh' or 'um'
    - is_valid_duration: at least `m` seconds long
    - is_pre_pause: at least `n` seconds of silence following the filler
    - is_single_speaker: at least `s` seconds away from other speaker activity (A non confusing hold)
    - FIELDS:
        - utt_idx
        - speaker
        - utt_start
        - utt_end
        - filler_start
        - filler_end
        - text
        - words
        - word_starts
        - word_ends
2. Extract all "good shifts" qy
    - 

"""

FILLER_CANDS = ["uh", "um"]
ANNO_PATH = "data/swb_ms98_transcriptions"
DA_PATH = "data/swb_dialog_acts_words"
TRAIN = "data/splits/train.txt"
VAL = "data/splits/val.txt"
TEST = "data/splits/test.txt"
CHECKPOINT = "data/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
AUDIO_REL_PATH = "data/relative_audio_path.json"
# WARNING: make sure you use the correct audio-root
AUDIO_ROOT = "/home/erik/projects/data/switchboard/audio"


def load_dataframe(path):
    def _times(x):
        return json.loads(x)

    def _text(x):
        return [a[1:-1] for a in x.strip("][").split(", ")]

    converters = {
        "starts": _times,
        "ends": _times,
        "da": _text,
        "da_boi": _text,
        "words": _text,
    }
    return pd.read_csv(path, converters=converters)


def is_filler(w):
    return w in FILLER_CANDS


def is_valid_duration(ii, utt, min_dur=0.2):
    if utt["ends"][ii] - utt["starts"][ii] >= min_dur:
        return True
    return False


def is_pre_pause(wii, utt, next_utt_start, min_pause=0.1):
    filler_end = utt["ends"][wii]
    if wii < len(utt["starts"]) - 1:
        next_start = utt["starts"][wii + 1]
    else:
        next_start = next_utt_start
    if next_start - filler_end >= min_pause:
        return True
    return False


def is_isolated(wii, utt, other_utts, min_isolation=1.0):
    """
    condition that checks if the other speaker is active close to the filler
    """
    # Isolation region
    fstart = utt.starts[wii] - min_isolation
    fend = utt.ends[wii] + min_isolation

    # other speaker utterance start/end times
    other_start = other_utts["start"].to_numpy()
    other_end = other_utts["end"].to_numpy()

    for os, oe in zip(other_start, other_end):
        # Skip if other utterance ends before the isolated region
        if oe < fstart:
            # print('Continue: END is BEFORE isolated region START')
            continue

        # starts inside isolation-region
        if fstart <= os <= fend:
            # print('Starts in non-valid-region')
            return False

        # Ends inside isolation-region
        if fstart <= oe <= fend:
            # print('Ends in non-valid-region')
            return False

        # ^ above condtions covers utterances completely inside isolation-region
        # Spans across the region
        if os < fstart and fend < oe:
            # print('Spans entire region')
            return False

        if os > fend:
            # print('BREAK: START are AFTER isolated region END')
            break
    return True


def valid_qy(da):
    """last word in utterance is QY"""
    return da[-1] == "qy"


def valid_qy_post_pause(ii, df, min_intra_pause=0.5):
    if ii == len(df) - 1:
        # last entry
        return True

    next_utt_start = df.iloc[ii + 1]["start"]
    utt = df.iloc[ii]
    pause_dur = next_utt_start - utt.end
    if pause_dur < min_intra_pause:
        return False

    return True


def valid_qy_isolation(ii, df, other_df, min_isolation=0.5):
    utt = df.iloc[ii]
    region_start = utt["end"] - min_isolation
    region_end = utt["end"] + min_isolation
    other_start = other_df["start"]
    other_end = other_df["end"]
    for os, oe in zip(other_start, other_end):
        # Skip if other utterance ends before the isolated region
        if oe < region_start:
            # print('Continue: END is BEFORE isolated region START')
            continue

        # starts inside isolation-region
        if region_start <= os <= region_end:
            # print('Starts in non-valid-region')
            return False

        # Ends inside isolation-region
        if region_start <= oe <= region_end:
            # print('Ends in non-valid-region')
            return False

        # ^ above condtions covers utterances completely inside isolation-region
        # Spans across the region
        if os < region_start and region_end < oe:
            # print('Spans entire region')
            return False

        if region_end < os:
            # print('BREAK: START are AFTER isolated region END')
            break

    return True


def valid_qy_by_cross(crosses, max_survival=250):
    if crosses[0] == -1:
        return False

    if crosses[0] > max_survival:
        return False
    return True


def plot_filler_dur(df):
    df["duration"] = df["end"] - df["start"]
    fig, ax = plt.subplots(1, 1)
    df["duration"][df["filler"] == "uh"].hist(
        ax=ax, bins=100, range=(0.2, 1.2), color="g", alpha=0.5, label="uh"
    )
    df["duration"][df["filler"] == "um"].hist(
        ax=ax, bins=100, range=(0.2, 1.2), color="b", alpha=0.5, label="um"
    )
    ax.set_xlim([0.2, 1.2])
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("N")
    ax.legend()
    plt.show()


class SWBReader:
    def __init__(self, anno_path=ANNO_PATH, da_path=ANNO_PATH, valid_path=TEST):
        self.anno_path = anno_path
        self.da_path = da_path
        self.valid_sessions = read_txt(valid_path)
        self.valid_sessions.sort()
        self.session_to_path = self.get_session_paths()

    def get_session_paths(self):
        def _session_name(p):
            return (
                basename(p)
                .split("-")[0]
                .replace("sw", "")
                .replace("A", "")
                .replace("B", "")
            )

        files = glob(join(self.anno_path, "**/*A-ms98-a-trans.text"), recursive=True)
        files.sort()

        paths = {}
        for p in files:
            session = _session_name(p)
            if not session in self.valid_sessions:
                continue
            paths[session] = {
                "A": {
                    "trans": p,
                    "words": p.replace("A-ms98-a-trans.text", "A-ms98-a-word.text"),
                    "da_words": join(DA_PATH, f"sw{session}A-word-da.csv"),
                },
                "B": {
                    "trans": p.replace("A-ms98-a-trans.text", "B-ms98-a-trans.text"),
                    "words": p.replace("A-ms98-a-trans.text", "B-ms98-a-word.text"),
                    "da_words": join(DA_PATH, f"sw{session}B-word-da.csv"),
                },
            }
        return paths

    def read_utter_trans(self, path):
        """extract utterance annotation"""
        # trans = []
        trans = {}
        for row in read_txt(path):
            utt_idx, start, end, *text = row.split(" ")
            text = " ".join(text)
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]" or text == "[noise] [noise]":
                continue
            # trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
            trans[utt_idx] = {"start": start, "end": end, "text": text}
        return trans

    def read_word_trans(self, path):
        trans = []
        for row in read_txt(path):
            utt_idx, start, end, text = row.strip().split()
            start = float(start)
            end = float(end)
            if text == "[silence]":
                continue
            if text == "[noise]":
                continue
            trans.append({"utt_idx": utt_idx, "start": start, "end": end, "text": text})
        return trans

    def combine_utterance_and_words(
        self, speaker, words, utters, da_words, return_pandas=True
    ):
        utterances = []
        for utt_idx, utterance in utters.items():
            das = da_words[da_words["utt_idx"] == utt_idx]
            word_list, starts, ends = [], [], []
            for w in words:
                if (
                    utterance["end"] + 1 < w["start"]
                ):  # add slight extra padding to be sure
                    break
                if w["utt_idx"] == utt_idx:
                    word_list.append(w["text"])
                    starts.append(w["start"])
                    ends.append(w["end"])
            assert len(das) == len(
                word_list
            ), f"da-words don't match words {len(das)} != {len(words)}"

            utterance["speaker"] = speaker
            utterance["start"] = starts[0]
            utterance["end"] = ends[-1]
            utterance["starts"] = starts
            utterance["ends"] = ends
            utterance["words"] = word_list
            utterance["da"] = das["da"].to_list()
            utterance["da_boi"] = das["boi"].to_list()
            utterance["utt_idx"] = utt_idx
            utterances.append(utterance)

        if return_pandas:
            utterances = pd.DataFrame(utterances)
        return utterances

    def read_da_words(self, path):
        return pd.read_csv(
            path, names=["utt_idx", "start", "end", "word", "boi", "da", "da_idx"]
        )

    def get_session(self, session):
        p = self.session_to_path[session]
        A_utters = self.read_utter_trans(p["A"]["trans"])
        A_words = self.read_word_trans(p["A"]["words"])
        A_da_words = self.read_da_words(p["A"]["da_words"])
        B_utters = self.read_utter_trans(p["B"]["trans"])
        B_words = self.read_word_trans(p["B"]["words"])
        B_da_words = self.read_da_words(p["B"]["da_words"])
        info = {
            "A": self.combine_utterance_and_words("A", A_words, A_utters, A_da_words),
            "B": self.combine_utterance_and_words("B", B_words, B_utters, B_da_words),
        }
        return info

    def iter_sessions(self):
        for session in self.valid_sessions:
            yield session, self.get_session(session)


class Preprocess:
    def __init__(
        self,
        hop_time=0.01,
        f0_min=60,
        f0_max=400,
        min_prosody_dur=1,
        min_duration=0.2,
        min_pause=0.2,
        min_isolation=1.0,
        min_filler_utt_duration=1.0,
        sample_rate=8000,
        audio_root=AUDIO_ROOT,
    ):
        self.reader = SWBReader()
        self.audio_root = audio_root
        self.relative_audio_path = read_json(AUDIO_REL_PATH)

        # Prosody
        self.sample_rate = sample_rate
        self.hop_time = hop_time
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.min_prosody_dur = min_prosody_dur

        # fillers
        self.min_duration = min_duration
        self.min_pause = min_pause
        self.min_isolation = min_isolation
        self.min_filler_utt_duration = min_filler_utt_duration

    def torch_to_praat_sound(self, x: torch.Tensor) -> parselmouth.Sound:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().astype("float64")
        return parselmouth.Sound(x, sampling_frequency=self.sample_rate)

    def praat_to_torch(self, sound: parselmouth.Sound) -> torch.Tensor:
        y = sound.as_array().astype("float32")
        return torch.from_numpy(y)

    def segment_prosody(self, waveform: torch.Tensor):
        sound = self.torch_to_praat_sound(waveform)

        # F0
        pitch = sound.to_pitch(
            time_step=self.hop_time, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max
        )
        intensity = sound.to_intensity(
            time_step=self.hop_time,
            minimum_pitch=self.f0_min,
            subtract_mean=True,  # subtract_mean is True by default
        )

        # To tensor
        pitch = torch.from_numpy(pitch.selected_array["frequency"]).float()
        intensity = self.praat_to_torch(intensity)[0]
        return pitch, intensity

    def time_to_sample(self, t):
        return round(t * self.sample_rate)

    def time_to_hop_frame(self, t):
        return round(t / self.hop_time)

    def get_session_df(self, session, utt_df):
        sdf = utt_df[utt_df["session"] == session]
        return {"A": sdf[sdf["speaker"] == "A"], "B": sdf[sdf["speaker"] == "B"]}

    def extract_session_prosody(self, session, info):
        """
        Add prosody metrics to utterance and find global stats
        """

        audio_path = join(self.audio_root, self.relative_audio_path[session] + ".wav")
        waveform, _ = load_waveform(audio_path, sample_rate=self.sample_rate)

        new_df = []
        global_prosody = {}
        for speaker, df in info.items():
            channel = 0 if speaker == "A" else 1
            channel_pitch = []
            channel_ints = []
            for ii in range(len(df)):
                utt = df.iloc[ii].copy()
                duration = utt["end"] - utt["start"]
                if duration < self.min_prosody_dur:
                    utt["utt_f0_m"] = -1
                    utt["utt_f0_s"] = -1
                    utt["utt_intensity_m"] = -1
                    utt["utt_intensity_s"] = -1
                    new_df.append(utt)
                else:
                    sample_start = self.time_to_sample(utt["start"])
                    sample_end = self.time_to_sample(utt["end"])
                    wav_tmp = waveform[channel, sample_start:sample_end]
                    tmp_f0, tmp_int = self.segment_prosody(wav_tmp)
                    voice = tmp_f0[tmp_f0 != 0]
                    channel_pitch.append(voice)
                    channel_ints.append(tmp_int)
                    utt["utt_f0_m"] = voice.mean().item()
                    utt["utt_f0_s"] = voice.std().item()
                    utt["utt_intensity_m"] = tmp_int.mean().item()
                    utt["utt_intensity_s"] = tmp_int.std().item()
                    new_df.append(utt)
            global_prosody[speaker] = {
                "f0": torch.cat(channel_pitch),
                "intensity": torch.cat(channel_ints),
            }

        ndf = pd.DataFrame(new_df)

        # Global stats speaker A
        ndf.loc[ndf["speaker"] == "A", "speaker_f0_m"] = (
            global_prosody["A"]["f0"].mean().item()
        )
        ndf.loc[ndf["speaker"] == "A", "speaker_f0_s"] = (
            global_prosody["A"]["f0"].std().item()
        )
        ndf.loc[ndf["speaker"] == "A", "speaker_intensity_m"] = (
            global_prosody["A"]["intensity"].mean().item()
        )
        ndf.loc[ndf["speaker"] == "A", "speaker_intensity_m"] = (
            global_prosody["A"]["intensity"].std().item()
        )
        # Global stats speaker B
        ndf.loc[ndf["speaker"] == "B", "speaker_f0_m"] = (
            global_prosody["B"]["f0"].mean().item()
        )
        ndf.loc[ndf["speaker"] == "B", "speaker_f0_s"] = (
            global_prosody["B"]["f0"].std().item()
        )
        ndf.loc[ndf["speaker"] == "B", "speaker_intensity_m"] = (
            global_prosody["B"]["intensity"].mean().item()
        )
        ndf.loc[ndf["speaker"] == "B", "speaker_intensity_m"] = (
            global_prosody["B"]["intensity"].std().item()
        )

        # Add session
        ndf["session"] = session
        return ndf

    def extract_filler_prosody(self, wii, utt, waveform):
        channel = 0 if utt["speaker"] == "A" else 1

        sample_start = self.time_to_sample(utt["start"])
        sample_end = self.time_to_sample(utt["end"])
        wav_tmp = waveform[channel, sample_start:sample_end]

        utt_f0, utt_ints = self.segment_prosody(wav_tmp)

        # Pad to same size
        diff = len(utt_f0) - len(utt_ints)
        if diff > 0:
            # this seems to match with pitch data
            pre_pad = torch.empty(diff).fill_(utt_ints[0])
            utt_ints = torch.cat([pre_pad, utt_ints])

        # Extract only prosody over filler
        rel_filler_start = utt["starts"][wii] - utt["start"]
        rel_filler_end = utt["ends"][wii] - utt["start"]
        fstart = self.time_to_hop_frame(rel_filler_start)
        fend = self.time_to_hop_frame(rel_filler_end)

        f0 = utt_f0[fstart:fend]
        ints = utt_ints[fstart:fend]
        f0m = f0[f0 != 0].mean().item()
        f0s = f0[f0 != 0].std().item()
        inm = ints.mean().item()
        ins = ints.std().item()

        # fm = utt_f0[utt_f0 != 0].mean().item()
        # im = utt_ints.mean().item()
        # print("utt_f0: ", tuple(utt_f0.shape))
        # print("utt_ints: ", tuple(utt_ints.shape))
        # print("fstart: ", fstart, rel_filler_start)
        # print("fend: ", fend, rel_filler_end)
        #
        # print("f0m: ", f0m, fm)
        # print("f0s: ", f0s)
        # print("inm: ", inm, im)
        # print("ins: ", ins)
        # input()
        return f0m, f0s, inm, ins

    def extract_session_fillers(
        self,
        session,
        info,
        min_duration=0.2,
        min_pause=0.2,
        min_isolation=1.0,
    ):
        audio_path = join(
            self.audio_root, self.relative_audio_path[str(session)] + ".wav"
        )
        waveform, _ = load_waveform(audio_path, sample_rate=self.sample_rate)

        condition = {
            "duration": 0,
            "pre_pause": 0,
            "isolated": 0,
        }
        fillers = []

        for speaker in ["A", "B"]:
            other_speaker = "A" if speaker == "B" else "B"
            channel = 0 if speaker == "A" else 1

            # break
            for ii in range(len(info[speaker])):
                utt = info[speaker].iloc[ii]
                if utt["end"] - utt["start"] < self.min_filler_utt_duration:
                    continue

                # Get next utterance start to check pre-pause-condition
                # if its the last entry we assume a very large value until the next start
                if ii < len(info[speaker]) - 1:
                    next_utt_start = info[speaker].iloc[ii + 1]["starts"][0]
                else:
                    next_utt_start = 9999999999
                # Loop over all the words to find a filler
                for wii, w in enumerate(utt["words"]):
                    if not is_filler(w):
                        continue
                    if not is_valid_duration(wii, utt, min_dur=min_duration):
                        condition["duration"] += 1
                        continue
                    if not is_pre_pause(wii, utt, next_utt_start, min_pause=min_pause):
                        condition["pre_pause"] += 1
                        continue
                    if not is_isolated(
                        wii, utt, info[other_speaker], min_isolation=min_isolation
                    ):
                        condition["isolated"] += 1
                        continue

                    # Extract filler Prosody
                    f0m, f0s, inm, ins = self.extract_filler_prosody(wii, utt, waveform)

                    fillers.append(
                        {
                            "session": session,
                            "speaker": speaker,
                            "filler": w,
                            "start": utt["starts"][wii],
                            "end": utt["ends"][wii],
                            "da": utt["da"][wii],
                            "filler_f0_m": f0m,
                            "filler_f0_s": f0s,
                            "filler_intensity_m": inm,
                            "filler_intensity_s": ins,
                            "utt_idx": utt["utt_idx"],
                            "utt_start": utt["starts"][0],
                            "utt_end": utt["ends"][-1],
                            "utt_loc": wii,
                            "utt_n_words": len(utt["words"]),
                            "utt_f0_m": utt["utt_f0_m"],
                            "utt_f0_s": utt["utt_f0_m"],
                            "utt_intensity_m": utt["utt_intensity_m"],
                            "utt_intensity_s": utt["utt_intensity_m"],
                            "speaker_f0_m": utt["speaker_f0_m"],
                            "speaker_f0_s": utt["speaker_f0_m"],
                            "speaker_intensity_m": utt["speaker_intensity_m"],
                            "speaker_intensity_s": utt["speaker_intensity_m"],
                        }
                    )
        return fillers, condition

    def create_utterance_dataframe(self, max_sessions=-1):
        all_df = []
        for ii, (session, info) in tqdm(
            enumerate(self.reader.iter_sessions()),
            total=len(self.reader.valid_sessions),
            desc="Preprocess",
        ):
            df = self.extract_session_prosody(session, info)
            all_df.append(df)
            if ii == max_sessions:
                print("Maximum sessions reached. ", max_sessions)
                break
        dff = pd.concat(all_df)
        return dff

    def create_filler_dataframe(self, utt_df):
        fillers = []
        condition = {"duration": 0, "pre_pause": 0, "isolated": 0}
        for session in tqdm(utt_df["session"].unique(), desc="Fillers"):
            info = self.get_session_df(session, utt_df)
            tmp_fillers, conds = self.extract_session_fillers(
                session,
                info,
                min_duration=self.min_duration,
                min_pause=self.min_pause,
                min_isolation=self.min_isolation,
            )
            fillers += tmp_fillers
            for cond, n in conds.items():
                condition[cond] += n

        print("min_duration: ", self.min_duration)
        print("min_pause: ", self.min_pause)
        print("min_isolation: ", self.min_isolation)
        print("Valid Fillers: ", len(fillers))
        tot = 0
        for c, n in condition.items():
            print(f"{c}: {n}")
            tot += n
        print("Non Valid: ", tot)
        return pd.DataFrame(fillers)


class Experiment:
    def __init__(
        self,
        silence=10,
        context=20,
        max_batch=5,
        qy_max_shift_time=5,
        utt_path="data/utterances.csv",
        fill_path="data/fillers.csv",
        checkpoint=CHECKPOINT,
        audio_root=AUDIO_ROOT,
    ):
        self.checkpoint = checkpoint
        self.model = None  # self.load_model(checkpoint)
        self.max_batch = max_batch
        self.sample_rate = 16_000
        self.frame_hz = 50
        self.reader = SWBReader()

        self.silence = silence
        self.context = context
        self.qy_max_shift_frames = round(self.frame_hz * qy_max_shift_time)

        self.audio_root = audio_root
        self.relative_audio_path = read_json(AUDIO_REL_PATH)
        self.utt_df = load_dataframe(utt_path)
        self.fill_df = load_dataframe(fill_path)

    def load_model(self):
        print("Load Model...")
        model = VAPModel.load_from_checkpoint(self.checkpoint)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")

        self.model = model
        self.sample_rate = model.sample_rate
        self.frame_hz = model.frame_hz

    def load_filler_audio(self, f):
        speaker_idx = 0 if f.speaker == "A" else 1
        other_idx = 0 if speaker_idx == 1 else 1

        y, _ = load_waveform(
            join(self.audio_root, self.relative_audio_path[str(f.session)] + ".wav"),
            start_time=f.start,
            end_time=f.end,
            sample_rate=self.sample_rate,
        )
        y[other_idx].fill_(0.0)
        return y.unsqueeze(0)  # add batch dim

    def pad_silence(self, waveform, silence=10, sil_samples=None):
        assert (
            waveform.ndim == 3
        ), f"Expects waveform of shape (B, C, n_samples) but got {waveform.shape}"
        if sil_samples is None:
            sil_samples = int(silence * self.sample_rate)
        B, C, _ = waveform.size()
        z = torch.zeros((B, C, sil_samples), device=waveform.device)
        return torch.cat([waveform, z], dim=-1)

    def to_samples(self, t):
        return round(t * self.sample_rate)

    def to_frames(self, t):
        return round(t * self.frame_hz)

    def find_cross(self, p, cross_frames=1, cutoff=0.5):
        assert p.ndim == 1, f"Expects (n_frames,) got {p.shape}"
        cross_idx = torch.where(p <= cutoff)[0]
        cross = -1
        if len(cross_idx) > 0:
            cross = cross_idx[0].cpu().item()

        p = p_now[fill_idx, 1000:]
        cutoff = 0.5
        cross_frames = 3
        cross_idx = torch.where(p <= cutoff)[0]
        if cross_frames > 1:
            cuf = cross_idx.unfold(0, cross_frames, step=1)
            diffs = (cuf[:, 1:] - cuf[:, :-1]).sum(-1) == (cross_frames - 1)
            idx = torch.where(diffs)[0][0]
            cross = cross_idx[idx].item()
        else:
            cross = cross_idx[0].item()

        return cross

    def get_filler_omission_sample(self, filler):
        """
        1. Load the "full" audio (i.e. with the end filler and context).
        2. Remove the filler samples to create `omission` audio
        3. Add silence to the longest (including the filler) audio
        4. Add silence frames to the 'omission' audio to same size
        5. Batch and return

        Returns:
            x:                      torch.tensor (2, 2, N_SAMPLES) batch of filler/omission
            relative_filler_start:  float, time of relative filler start
            relative_filler_end:    float, time of relative filler end
        """
        ##################################################
        # Frames and samples
        ##################################################
        t0 = filler.start - self.context
        t1 = filler.start
        t2 = filler.end

        if t0 < 0:
            t0 = 0

        ##################################################
        # Load audio
        ##################################################
        audio_path = join(
            self.audio_root, self.relative_audio_path[str(filler.session)] + ".wav"
        )
        waveform, _ = load_waveform(
            audio_path,
            start_time=t0,
            end_time=t2,
            sample_rate=self.sample_rate,
        )
        waveform = waveform.unsqueeze(0)

        ##################################################
        # Omit filler
        ##################################################
        filler_dur = t2 - t1
        filler_dur_samples = self.to_samples(filler_dur)
        wav_omission = waveform[
            ..., :-filler_dur_samples
        ]  # omit last frames containing filler

        # Add silence
        # create batch
        wav_filler = self.pad_silence(waveform, silence=self.silence)
        diff_samples = wav_filler.shape[-1] - wav_omission.shape[-1]
        wav_omission = self.pad_silence(wav_omission, sil_samples=diff_samples)

        rel_filler_start = t1 - t0
        rel_filler_end = t2 - t0
        return torch.cat([wav_filler, wav_omission]), rel_filler_start, rel_filler_end

    def get_filler_omission_cross(self, filler, out, rel_filler_start, rel_filler_end):
        speaker_idx = 0 if filler.speaker == "A" else 1

        filler_end_frame = round(rel_filler_end * self.frame_hz)
        filler_start_frame = round(rel_filler_start * self.frame_hz)

        # Extract crosses
        p_now_filler = out["p_now"][0, filler_end_frame:, speaker_idx]
        p_fut_filler = out["p_future"][0, filler_end_frame:, speaker_idx]
        filler_now_cross = exp.find_cross(p_now_filler, cross_frames=1)
        filler_fut_cross = exp.find_cross(p_fut_filler, cross_frames=1)

        p_now_omission = out["p_now"][1, filler_start_frame:, speaker_idx]
        p_fut_omission = out["p_future"][1, filler_start_frame:, speaker_idx]
        omission_now_cross = exp.find_cross(p_now_omission, cross_frames=1)
        omission_fut_cross = exp.find_cross(p_fut_omission, cross_frames=1)

        return {
            "filler_now_cross": filler_now_cross,
            "filler_fut_cross": filler_fut_cross,
            "omit_now_cross": omission_now_cross,
            "omit_fut_cross": omission_fut_cross,
        }

    @torch.no_grad()
    def experiment_omit_fillers(self, max_rows=-1):
        new_df = []

        if self.model is None:
            self.load_model()

        N = len(self.fill_df)
        for ii, row_idx in enumerate(tqdm(list(range(N)), desc="Omit Fillers")):
            if ii == max_rows:
                break

            filler = self.fill_df.loc[row_idx]
            if filler.start < self.context:
                continue

            x, rel_filler_start, rel_filler_end = self.get_filler_omission_sample(
                filler
            )
            out = self.model.probs(x.to(self.model.device))
            crosses = self.get_filler_omission_cross(
                filler, out, rel_filler_start, rel_filler_end
            )

            # Create new extended row
            new_filler = filler.to_dict()
            new_filler["filler_now_cross"] = crosses["filler_now_cross"]
            new_filler["filler_fut_cross"] = crosses["filler_fut_cross"]
            new_filler["omit_now_cross"] = crosses["omit_now_cross"]
            new_filler["omit_fut_cross"] = crosses["omit_fut_cross"]
            new_df.append(new_filler)
        return pd.DataFrame(new_df)

    def get_filler_addition_sample(self, utt, spkr_fillers):
        session = utt.session
        start = utt.end - self.context

        string_session = session
        if isinstance(string_session, pd.Series):
            string_session = str(int(string_session))
        else:  # isinstance(string_session, int):
            string_session = str(string_session)

        waveform, _ = load_waveform(
            join(
                self.audio_root,
                self.relative_audio_path[string_session] + ".wav",
            ),
            start_time=start,
            end_time=utt.end,
            sample_rate=self.sample_rate,
        )
        waveform = waveform.unsqueeze(0)

        # Audio for QY without filler
        sil_start_time = [self.context]
        audio = [waveform[0].permute(1, 0)]
        for fidx in range(len(spkr_fillers)):
            f = spkr_fillers.iloc[fidx]
            filler_dur = f.end - f.start

            # This sample ends after the context and the filler duration
            sil_start_time.append(self.context + filler_dur)

            # Add filler directly after 'qy'
            wfill = self.load_filler_audio(f)
            wfill = torch.cat((waveform, wfill), dim=-1)
            x_tmp = self.pad_silence(wfill, silence=self.silence)
            audio.append(x_tmp[0].permute(1, 0))

        sil_start_frames = (torch.tensor(sil_start_time) * self.frame_hz).round().long()
        x = pad_sequence(audio, batch_first=True).permute(0, 2, 1)
        return x, sil_start_frames

    def get_qy_speaker_fillers(self, speaker, session):
        sdf = exp.fill_df[exp.fill_df["session"] == session]
        return sdf[sdf["speaker"] == speaker]

    def get_batch_crosses(
        self, x, sil_start_frames, channel, cross_frames=1, include_probs=False
    ):
        now_crosses, fut_crosses = [], []
        all_p_now, all_p_fut = [], []

        if x.shape[0] <= self.max_batch:
            out = self.model.probs(x.to(self.model.device))
            for i in range(out["p_now"].shape[0]):
                p_now = out["p_now"][i, :, channel].cpu()
                p_fut = out["p_future"][i, :, channel].cpu()
                sil_start = sil_start_frames[i]

                if include_probs:
                    all_p_now.append(p_now)
                    all_p_fut.append(p_fut)

                now_cross_idx = self.find_cross(p_now[sil_start:], cross_frames)
                fut_cross_idx = self.find_cross(p_fut[sil_start:], cross_frames)
                now_crosses.append(now_cross_idx)
                fut_crosses.append(fut_cross_idx)
        else:

            ii = 0
            # split to maximum batch-size
            for xx in x.split(self.max_batch, dim=0):
                out = self.model.probs(xx.to(self.model.device))
                n = len(xx)

                chunk_sil_starts = sil_start_frames[range(ii, ii + n)]

                for jj in range(out["p_now"].shape[0]):
                    p_now = out["p_now"][jj, :, channel].cpu()
                    p_fut = out["p_future"][jj, :, channel].cpu()
                    sil_start = chunk_sil_starts[jj]

                    if include_probs:
                        all_p_now.append(p_now)
                        all_p_fut.append(p_fut)

                    now_cross_idx = self.find_cross(p_now[sil_start:], cross_frames)
                    fut_cross_idx = self.find_cross(p_fut[sil_start:], cross_frames)
                    now_crosses.append(now_cross_idx)
                    fut_crosses.append(fut_cross_idx)
                ii += n

        if include_probs:
            all_p_now = torch.stack(all_p_now)
            all_p_fut = torch.stack(all_p_fut)
            return now_crosses, fut_crosses, all_p_now, all_p_fut

        return now_crosses, fut_crosses

    @torch.no_grad()
    def experiment_qy_fillers(self, max_entries=-1):
        if self.model == None:
            self.load_model()

        def get_session_df(session, utt_df):
            sdf = utt_df[utt_df["session"] == session]
            return {"A": sdf[sdf["speaker"] == "A"], "B": sdf[sdf["speaker"] == "B"]}

        min_isolation = 0.5
        questions = []
        condition = {
            "no_filler": 0,
            "no_context": 0,
            "no_shift": 0,
            "post_pause": 0,
            "isolation": 0,
        }
        stop = False
        for session in tqdm(self.utt_df["session"].unique(), desc="QY"):
            info = get_session_df(session, self.utt_df)

            for speaker in ["A", "B"]:
                other_speaker = "A" if speaker == "B" else "B"
                df = info[speaker]
                other_df = info[other_speaker]
                fillers = self.get_qy_speaker_fillers(speaker, session)

                # Requires at least one filler to add
                if len(fillers) == 0:
                    condition["no_filler"] += 1
                    continue

                # Iterate over utterances and find Question-Yes dialog acts
                for ii, da in enumerate(df["da"]):

                    utt = df.iloc[ii]
                    start = utt.end - self.context

                    # Require enough context
                    if start < 0:
                        condition["no_context"] += 1
                        continue

                    # Requires that last dialog act in utterance is QY
                    if not valid_qy(da):
                        continue

                    # Requires that the current speaker actually takes a pause
                    if not valid_qy_post_pause(ii, df):
                        condition["post_pause"] += 1
                        continue

                    # Requires that the "listener" is not speaking close to the
                    # filler.
                    if not valid_qy_isolation(ii, df, other_df, min_isolation):
                        condition["isolation"] += 1
                        continue

                    # Get model outputs
                    channel = 0 if utt.speaker == "A" else 1
                    x, sil_start_frames = self.get_filler_addition_sample(utt, fillers)
                    now_crosses, fut_crosses = self.get_batch_crosses(
                        x, sil_start_frames, channel
                    )

                    # Check if QY was considered a shift by the model
                    # A shift must be detected inside a region of `self.qy_max_shift_frames`
                    # from the end of the QY
                    if not valid_qy_by_cross(
                        now_crosses, max_survival=self.qy_max_shift_frames
                    ):
                        condition["no_shift"] += 1
                        continue

                    # TODO: what values are required for this experiment?
                    # Add values

                    qq = utt.copy()
                    qq["q_idx"] = utt.utt_idx
                    qq["q_is_filler"] = 0
                    qq["q_now_cross"] = now_crosses[0]
                    qq["q_fut_cross"] = fut_crosses[0]
                    questions.append(qq)

                    for i in range(1, len(now_crosses)):
                        fidx = i - 1
                        ff = fillers.iloc[fidx].copy()
                        ff["q_idx"] = utt.utt_idx
                        ff["q_is_filler"] = 1
                        ff["q_now_cross"] = now_crosses[i]
                        ff["q_fut_cross"] = fut_crosses[i]
                        questions.append(ff)

                    if max_entries > 0 and len(questions) >= max_entries:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        print("QY: ", len(questions))
        print("Condition")
        for k, v in condition.items():
            print(f"{k}: {v}")

        return pd.DataFrame(questions)


class Result:
    @staticmethod
    def survival_omission(df, max_silence_frames=500):
        """
        filler_now_survival

        survival_time
        """

        def get_log_rank_params(filler_event, filler_exit, omit_event, omit_exit):
            lf = []
            for st, su in zip(filler_event.values, filler_exit.values):
                lf.append((st, su))
            yf = np.array(lf, dtype=[("name", "?"), ("age", "i8")])
            lo = []
            for st, su in zip(omit_event.values, omit_exit.values):
                lo.append((st, su))
            yo = np.array(lo, dtype=[("name", "?"), ("age", "i8")])
            y = np.concatenate((yf, yo))
            group = ["filler"] * len(yf) + ["omit"] * len(yo)
            return y, group

        # filler
        filler_exit = df["filler_now_cross"].copy()
        filler_event = filler_exit > 0
        filler_exit[filler_exit < 0] = max_silence_frames
        # omission
        omit_exit = df["omit_now_cross"].copy()
        omit_event = omit_exit > 0
        omit_exit[omit_exit < 0] = max_silence_frames

        # Survival plot
        t_omit, p_omit = kaplan_meier_estimator(event=omit_event, time_exit=omit_exit)
        t_filler, p_filler = kaplan_meier_estimator(
            event=filler_event, time_exit=filler_exit
        )

        # Significance
        y, group = get_log_rank_params(filler_event, filler_exit, omit_event, omit_exit)
        chisq, pvalue, stats, covariance = compare_survival(y, group, return_stats=True)
        Result.plot_survival([t_omit, p_omit, t_filler, p_filler], chisq, pvalue)

        return {
            "time": {"omit": t_omit, "filler": t_filler},
            "survival_probs": {"omit": p_omit, "filler": p_filler},
            "chisq": chisq,
            "pvalue": pvalue,
        }

    @staticmethod
    def survival_qy(df, max_silence_frames=250):
        Q = df[df["q_is_filler"] == 0]
        Q.loc[:, "event"] = True
        Q.loc[Q["q_now_cross"] < 0, "event"] = False
        Q.loc[Q["q_now_cross"] < 0, "q_now_cross"] = max_silence_frames

        F = df[df["q_is_filler"] == 1]
        F.loc[:, "event"] = True
        F.loc[F["q_now_cross"] < 0, "event"] = False
        F.loc[F["q_now_cross"] < 0, "q_now_cross"] = max_silence_frames

        # Survival plot
        t_qy, p_qy = kaplan_meier_estimator(event=Q["event"], time_exit=Q["q_now_cross"])
        t_qy_filler, p_qy_filler = kaplan_meier_estimator(
            event=F["event"], time_exit=F["q_now_cross"]
        )
        Result.plot_survival(
            [t_qy, p_qy, t_qy_filler, p_qy_filler], title="QY + Fillers"
        )
        return {
            "time": {"qy": t_qy, "filler": t_qy_filler},
            "probs": {"qy": p_qy, "filler": p_qy_filler},
        }

    @staticmethod
    def plot_survival(
        TP, chisq=None, pvalue=None, silence=10, title="Filler and Omission"
    ):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        if chisq is None:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{title}\nChisq={round(chisq, 1)}, P={pvalue}")

        t1 = TP[0] / 50
        t2 = TP[2] / 50
        ax.step(t2, TP[3], where="post", color="r", label="Filler")
        ax.step(t1, TP[1], where="post", color="teal", label="Omit")
        ax.set_ylabel("est. probability of shift")
        ax.set_xlabel("time s")
        ax.set_xlim([0, silence])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mel_spectrogram(
        y, ax, sample_rate=16_000, hop_time=0.02, frame_time=0.05, n_mels=80
    ):
        duration = y.shape[-1] / sample_rate
        xmin, xmax = 0, duration
        ymin, ymax = 0, 80

        hop_length = round(sample_rate * hop_time)
        frame_length = round(sample_rate * frame_time)
        spec = log_mel_spectrogram(
            y,
            n_mels=n_mels,
            n_fft=frame_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )
        ax[0].imshow(
            spec[0],
            interpolation="none",
            aspect="auto",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )
        ax[1].imshow(
            spec[1],
            interpolation="none",
            aspect="auto",
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
        )
        ax[0].set_yticks([])
        ax[1].set_yticks([])

    @staticmethod
    def plot_speaker_probs(x, p, ax, label="P", alpha=0.6, colors=["b", "orange"]):
        px = p - 0.5
        z = torch.zeros_like(p)
        ax.fill_between(
            x, px, z, where=px > 0, color=colors[0], label=f"A {label}", alpha=alpha
        )
        ax.fill_between(
            x, px, z, where=px < 0, color=colors[1], label=f"B {label}", alpha=alpha
        )
        ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([-0.25, 0.25])
        ax.set_yticklabels(["B", "A"])
        return ax

    @staticmethod
    def plot_vad(x, vad, ax, label=None):
        assert vad.ndim == 1, f"Expects (N_FRAMES, ) got {vad.shape}"
        ymin, ymax = ax.get_ylim()
        scale = ymax - ymin
        ax.plot(x, ymin + vad.cpu() * scale, color="w", label=label)

    @staticmethod
    def plot_one_dialog_two_predicions(
        waveform,
        p1,
        p2=None,
        vad=None,
        word_dict=None,
        plabel=["Filler", "Omission"],
        relative_start=0,
        sample_rate=16_000,
        frame_hz=50,
        fontsize=12,
        plot=False,
    ):
        assert (
            waveform.ndim == 2
        ), f"Expected waveform (C, N_SAMPLES) got {waveform.shape}"

        if p1 is not None:
            assert p1.ndim == 1, f"Expected p1 (N_FRAMES) got {p1.shape}"

        if p2 is not None:
            assert p2.ndim == 1, f"Expected p2 (N_FRAMES,) got {p2.shape}"

        time = torch.arange(len(p1)) / frame_hz

        fig, ax = plt.subplots(4, 1, sharex=True)
        Result.plot_mel_spectrogram(
            waveform, ax=ax[:2], sample_rate=sample_rate, hop_time=0.01
        )
        if vad is not None:
            Result.plot_vad(time, vad[:, 0], ax[0], label="VAD")
            Result.plot_vad(time, vad[:, 1], ax[1])

        if word_dict is not None:
            ymin, ymax = ax[0].get_ylim()
            span = ymax - ymin
            mid = ymin + span / 2
            for s, e, w in zip(
                word_dict["starts"], word_dict["ends"], word_dict["words"]
            ):
                x_mid = s + (e - s) / 2
                x_text = x_mid - relative_start
                ax[0].text(x_text, y=mid, s=w, fontsize=fontsize, color="w")

        # Model out
        Result.plot_speaker_probs(time, p1, ax[2], label=plabel[0])
        Result.plot_speaker_probs(time, p2, ax[3], label=plabel[1])

        for a in ax:
            a.legend()

        plt.subplots_adjust(
            left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0.05
        )
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def plot_experiment1_omission(
        waveform,
        p_filler,
        p_omission,
        speaker,
        cross_fill,
        cross_omit,
        rel_filler_start,
        rel_filler_end,
        filler,
        filler_start,
        plabel=["Filler", "Omission"],
        vad=None,
        word_dict=None,
        sample_rate=16_000,
        frame_hz=50,
        figsize=(12, 8),
        plot=False,
    ):
        assert (
            waveform.ndim == 2
        ), f"Expected waveform (C, N_SAMPLES) got {waveform.shape}"

        if p_filler is not None:
            assert (
                p_filler.ndim == 1
            ), f"Expected p_filler (N_FRAMES) got {p_filler.shape}"

        if p_omission is not None:
            assert (
                p_omission.ndim == 1
            ), f"Expected p_omission (N_FRAMES,) got {p_omission.shape}"

        time = torch.arange(len(p_filler)) / frame_hz
        channel = 0 if speaker == "A" else 1
        other_channel = 0 if channel == 1 else 1

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize)

        # Melspectrogram
        Result.plot_mel_spectrogram(
            waveform,
            ax=[ax[channel], ax[other_channel]],
            sample_rate=sample_rate,
            hop_time=0.01,
        )
        if vad is not None:
            Result.plot_vad(time, vad[:, channel], ax[0], label="VAD")
            Result.plot_vad(time, vad[:, other_channel], ax[1])

        # Words
        if word_dict is not None:
            t0 = filler_start - rel_filler_start
            ys = [50, 40, 30, 20, 10]
            n_rows = len(ys)

            jj = 0
            for s, e, w in zip(
                word_dict["starts"],
                word_dict["ends"],
                word_dict["words"],
            ):

                row_idx = jj % n_rows
                y = ys[row_idx]
                jj += 1
                rel_start = s - t0
                x_mid = rel_start + (e - s) / 2
                ax[0].text(
                    x=x_mid,
                    y=y,
                    s=w,
                    fontsize=14,
                    color="w",
                    rotation=20,
                    fontweight='bold',
                    horizontalalignment="center",
                )

        # Turn-shift probabilitiesout
        Result.plot_speaker_probs(
            time, p_filler, ax[2], label=plabel[0] + f" ({filler})"
        )
        Result.plot_speaker_probs(time, p_omission, ax[3], label=plabel[1])

        # Show filler boundaries
        ax[0].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[0].axvline(rel_filler_end, color="r")
        ax[1].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[1].axvline(rel_filler_end, color="r")
        ax[2].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[2].axvline(rel_filler_end, color="r")
        ax[3].axvline(rel_filler_start, color="r", linestyle="dashed")

        # Show cross duration arrows
        ############################################
        y_arrow = -0.25
        if cross_fill >= 0:
            x_start = rel_filler_end.item()
            dx = cross_fill / 50
            x_end = x_start + dx
            mid = x_start + dx / 2
            ax[-2].hlines(y=y_arrow, xmin=x_start, xmax=x_end, color="k")
            ax[-2].vlines(
                x=[x_start, x_end], ymin=y_arrow - 0.1, ymax=y_arrow + 0.1, color="k"
            )
            ax[-2].text(
                x=mid,
                y=y_arrow + 0.05,
                s=f"{round(dx, 2)}s",
                horizontalalignment="center",
            )
        ############################################
        if cross_omit >= 0:
            x_start = rel_filler_start.item()
            dx = cross_omit / 50
            x_end = x_start + dx
            mid = x_start + dx / 2
            ax[-1].hlines(y=y_arrow, xmin=x_start, xmax=x_end, color="k")
            ax[-1].vlines(
                x=[x_start, x_end], ymin=y_arrow - 0.1, ymax=y_arrow + 0.1, color="k"
            )
            ax[-1].text(
                x=mid,
                y=y_arrow + 0.05,
                s=f"{round(dx, 2)}s",
                horizontalalignment="center",
            )
        ############################################

        from_ax = 0
        if vad is None:
            from_ax = 2
        for a in ax[from_ax:]:
            a.legend()

        plt.subplots_adjust(
            left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0.05
        )
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def plot_experiment2_addition(
        waveform,
        p,
        p_addition,
        speaker,
        cross,
        cross_add,
        rel_filler_start,
        rel_filler_end,
        abs_end,
        filler,
        plabel=["QY", "QY+F"],
        vad=None,
        word_dict=None,
        sample_rate=16_000,
        frame_hz=50,
        plot=False,
        figsize=(12,8)
    ):
        """
        waveform:   waveform including the additive filler
        p:          probs of regular (no-filler) output
        p_addition: probs of added filler output
        speaker:    active speaker
        """
        assert (
            waveform.ndim == 2
        ), f"Expected waveform (C, N_SAMPLES) got {waveform.shape}"

        if p is not None:
            assert p.ndim == 1, f"Expected p (N_FRAMES) got {p.shape}"

        if p_addition is not None:
            assert (
                p_addition.ndim == 1
            ), f"Expected p_addition (N_FRAMES,) got {p_addition.shape}"

        if isinstance(rel_filler_start, torch.Tensor):
            rel_filler_start = rel_filler_start.item()

        if isinstance(rel_filler_end, torch.Tensor):
            rel_filler_end = rel_filler_end.item()

        time = torch.arange(len(p_addition)) / frame_hz
        channel = 0 if speaker == "A" else 1
        other_channel = 0 if channel == 1 else 1

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize)

        # MelSpectrogram
        Result.plot_mel_spectrogram(
            waveform,
            ax=[ax[channel], ax[other_channel]],
            sample_rate=sample_rate,
            hop_time=0.01,
        )
        if vad is not None:
            Result.plot_vad(time, vad[:, channel], ax[0], label="VAD")
            Result.plot_vad(time, vad[:, other_channel], ax[1])

        # Words
        if word_dict is not None:
            t0 = abs_end - rel_filler_start
            ys = [50, 40, 30, 20, 10]
            n_rows = len(ys)

            jj = 0
            for s, e, w in zip(
                word_dict["starts"],
                word_dict["ends"],
                word_dict["words"],
            ):
                row_idx = jj % n_rows
                y = ys[row_idx]
                jj += 1
                rel_start = s - t0
                x_mid = rel_start + (e - s) / 2
                ax[0].text(
                    x=x_mid,
                    y=y,
                    s=w,
                    fontsize=14,
                    color="w",
                    rotation=20,
                    fontweight='bold',
                    horizontalalignment="center",
                )

        # Model out
        Result.plot_speaker_probs(time, p, ax[2], label=plabel[0])
        Result.plot_speaker_probs(
            time, p_addition, ax[3], label=plabel[1] + f" ({filler})"
        )

        # Show filler boundaries
        ax[0].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[0].axvline(rel_filler_end, color="r")
        ax[1].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[1].axvline(rel_filler_end, color="r")
        ax[2].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[3].axvline(rel_filler_start, color="r", linestyle="dashed")
        ax[3].axvline(rel_filler_end, color="r")

        # Show cross duration arrows
        ############################################
        y_arrow = -0.25
        if cross >= 0:
            x_start = rel_filler_start
            dx = cross / 50
            x_end = x_start + dx
            mid = x_start + dx / 2
            ax[-2].hlines(y=y_arrow, xmin=x_start, xmax=x_end, color="k")
            ax[-2].vlines(
                x=[x_start, x_end], ymin=y_arrow - 0.1, ymax=y_arrow + 0.1, color="k"
            )
            ax[-2].text(
                x=mid,
                y=y_arrow + 0.05,
                s=f"{round(dx, 2)}s",
                horizontalalignment="center",
            )
        ############################################
        if cross_add >= 0:
            x_start = rel_filler_end
            dx = cross_add / frame_hz
            x_end = x_start + dx
            mid = x_start + dx / 2
            ax[-1].hlines(y=y_arrow, xmin=x_start, xmax=x_end, color="k")
            ax[-1].vlines(
                x=[x_start, x_end], ymin=y_arrow - 0.1, ymax=y_arrow + 0.1, color="k"
            )
            ax[-1].text(
                x=mid,
                y=y_arrow + 0.05,
                s=f"{round(dx, 2)}s",
                horizontalalignment="center",
            )
        ############################################

        from_ax = 0
        if vad is None:
            from_ax = 2
        for a in ax[from_ax:]:
            a.legend()

        plt.subplots_adjust(
            left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0.05
        )
        if plot:
            plt.pause(0.01)
        return fig, ax


def visualize_debuggin():
    """must play around somewhere"""

    ##########################################
    # Experiment 1. Omission
    ##########################################
    # Save some figure data for fast iteration
    # Experiment 1 visualization
    exp = Experiment()
    exp.load_model()
    om_df = pd.read_csv("results/exp1_omission.csv")

    for ii in range(len(om_df)):
        sample = om_df.iloc[ii]
        x, rel_filler_start, rel_filler_end = exp.get_filler_omission_sample(sample)
        out = exp.model.probs(x.to(exp.model.device))
        crosses = exp.get_filler_omission_cross(
            sample, out, rel_filler_start, rel_filler_end
        )
        if crosses["filler_now_cross"] == -1 and crosses["omit_now_cross"] == -1:
            continue
        ###########################################
        # Plot the output from the model
        ###########################################
        channel = 0 if sample.speaker == "A" else 1
        p_now = out["p_now"][..., channel].cpu()
        utt = exp.utt_df[exp.utt_df["utt_idx"] == sample.utt_idx].iloc[0]
        fig, ax = Result.plot_experiment1_omission(
            x[0],
            p_filler=p_now[0],
            p_omission=p_now[1],
            speaker=sample.speaker,
            rel_filler_start=rel_filler_start,
            rel_filler_end=rel_filler_end,
            cross_fill=crosses["filler_now_cross"],
            cross_omit=crosses["omit_now_cross"],
            filler=sample.filler,
            filler_start=sample.start,
            word_dict={
                "starts": utt.starts[: sample.utt_loc + 1],
                "ends": utt.ends[: sample.utt_loc + 1],
                "words": utt.words[: sample.utt_loc + 1],
            },
            figsize=(12, 8),
        )
        ax[-1].set_xlim([17, 30])
        ############################################
        plt.show()

    ##########################################
    # Experiment 2. Addition
    ##########################################
    exp = Experiment()
    exp.load_model()

    qy_df = load_dataframe("results/exp2_qy.csv")
    qy_df = load_dataframe("results/exp2_qy.csv")

    def _text(x):
        return [a[1:-1] for a in x.strip("][").split(", ")]

    converters = {
        "da": _text,
        "da_boi": _text,
        "words": _text,
    }
    df = pd.read_csv("results/exp2_qy.csv", converters=converters)

    Q = qy_df[qy_df["q_is_filler"] == 0]
    F = qy_df[qy_df["q_is_filler"] == 1]

    for ii in range(14, len(Q)):
        # Choose a QY utterance
        sample = Q.iloc[ii]
        # Find fillers used for that QY
        fillers = F[F["q_idx"] == sample.q_idx]
        x, sil_start_frames = exp.get_filler_addition_sample(sample, fillers)
        channel = 0 if sample.speaker == "A" else 1
        now_crosses, _, p_now, _ = exp.get_batch_crosses(
            x, sil_start_frames, channel, include_probs=True
        )
        print("p_now: ", tuple(p_now.shape))
        # Loop over all filler additions
        for fill_idx in range(1, len(p_now)):
            rel_filler_start = sil_start_frames[0].item() / exp.frame_hz
            rel_filler_end = sil_start_frames[fill_idx].item() / exp.frame_hz
            filler = fillers.iloc[fill_idx - 1]
            word_dict={
                "starts": json.loads(sample.starts),
                "ends": json.loads(sample.ends),
                "words": _text(sample.words),
            }
            fig, ax = Result.plot_experiment2_addition(
                x[fill_idx],
                p=p_now[0],
                p_addition=p_now[fill_idx],
                speaker=sample.speaker,
                cross=now_crosses[0],
                cross_add=now_crosses[fill_idx],
                rel_filler_start=20,
                rel_filler_end=rel_filler_end,
                abs_end=word_dict['ends'][-1],
                filler=filler.filler,
                word_dict=word_dict,
                figsize=(12, 8)
            )
            stime = word_dict['starts'][0] - word_dict['ends'][-1] + 20 - 0.2
            ax[-1].set_xlim([stime, 25])
            plt.show()


    return cross



    cross = -1
    if len(cross_idx) > 0:
        cross = cross_idx[0].cpu().item()


if __name__ == "__main__":
    print("VAP Filler Experiment")

    # Only required once (data is included)
    if not (exists("data/utterances.csv") and exists("data/fillers.csv")):
        P = Preprocess()
        utt_df = P.create_utterance_dataframe()
        utt_df.to_csv("data/utterances.csv", index=False)
        fill_df = P.create_filler_dataframe(utt_df)
        fill_df.to_csv("data/fillers.csv", index=False)

    # Experiment 1
    if exists("results/exp1_omission.csv"):
        omission_df = pd.read_csv("results/exp1_omission.csv")
    else:
        print("Experiment 1")
        exp = Experiment()
        omission_df = exp.experiment_omit_fillers()
        omission_df.to_csv("results/exp1_omission.csv", index=False)

    experiment1_out = Result.survival_omission(omission_df)

    # Experiment 2
    if exists("results/exp2_qy.csv"):
        qy_df = pd.read_csv("results/exp2_qy.csv")
    else:
        print("Experiment 2")
        exp = Experiment()
        qy_df = exp.experiment_qy_fillers()
        qy_df.to_csv("results/exp2_qy.csv", index=False)

    experiment2_out = Result.survival_qy(qy_df)
