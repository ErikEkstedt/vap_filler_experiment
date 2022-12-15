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
from vap.audio import load_waveform


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
        utt_path="data/utterances.csv",
        fill_path="data/fillers.csv",
        checkpoint=CHECKPOINT,
        audio_root=AUDIO_ROOT,
    ):
        self.checkpoint = checkpoint
        self.model = None  # self.load_model(checkpoint)
        self.sample_rate = 16_000
        self.frame_hz = 50
        self.reader = SWBReader()

        self.silence = silence
        self.context = context

        self.audio_root = audio_root
        self.relative_audio_path = read_json(AUDIO_REL_PATH)
        self.utt_df = load_dataframe(utt_path)
        self.fill_df = load_dataframe(fill_path)

    def load_model(self, checkpoint):
        print("Load Model...")
        model = VAPModel.load_from_checkpoint(checkpoint)
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

    def find_cross(self, p, cutoff=0.5):
        assert p.ndim == 1, f"Expects (n_frames,) got {p.shape}"
        cross_idx = torch.where(p <= cutoff)[0]
        cross = -1
        if len(cross_idx) > 0:
            cross = cross_idx[0].cpu().item()
        return cross

    def omit_fillers(self, max_rows=-1):
        new_df = []

        if self.model is None:
            self.load_model(self.checkpoint)

        N = len(self.fill_df)
        for ii, row_idx in enumerate(tqdm(list(range(N)), desc="Omit Fillers")):
            if ii == max_rows:
                break

            filler = self.fill_df.loc[row_idx]
            if filler.start < self.context:
                continue

            ##################################################
            # Frames and samples
            ##################################################
            start = filler.start - self.context
            filler_dur = filler.end - filler.start

            filler_samples = self.to_samples(filler_dur)
            filler_frames = self.to_frames(filler_dur)
            sil_start_omit = self.to_frames(self.context)
            sil_start_filler = sil_start_omit + filler_frames

            ##################################################
            # Load audio
            ##################################################
            audio_path = join(
                self.audio_root, self.relative_audio_path[str(filler.session)] + ".wav"
            )
            waveform, _ = load_waveform(
                audio_path,
                start_time=start,
                end_time=filler.end,
                sample_rate=self.sample_rate,
            )
            waveform = waveform.unsqueeze(0)
            wav_filler = self.pad_silence(waveform, silence=self.silence)

            ##################################################
            # Combine filler/omit filler
            ##################################################
            wav_omit = waveform[..., :-filler_samples]
            diff = wav_filler.shape[-1] - wav_omit.shape[-1]
            wav_omit = self.pad_silence(wav_omit, sil_samples=diff)

            ##################################################
            # Forward
            ##################################################
            y = torch.cat([wav_filler, wav_omit])
            out = self.model.probs(y.to(self.model.device))

            # Extract crosses
            speaker_idx = 0 if filler.speaker == "A" else 1
            pnf = out["p_now"][0, sil_start_filler:, speaker_idx]
            pff = out["p_now"][0, sil_start_filler:, speaker_idx]
            pno = out["p_now"][1, sil_start_omit:-filler_frames, speaker_idx]
            pfo = out["p_now"][1, sil_start_omit:-filler_frames, speaker_idx]

            new_filler = filler.to_dict()
            new_filler["filler_now_cross"] = self.find_cross(pnf)
            new_filler["filler_fut_cross"] = self.find_cross(pff)
            new_filler["omit_now_cross"] = self.find_cross(pno)
            new_filler["omit_fut_cross"] = self.find_cross(pfo)
            new_df.append(new_filler)
        return pd.DataFrame(new_df)

    @torch.no_grad()
    def experiment_qy(self, max_entries=-1, max_batch=5):
        if self.model == None:
            self.load_model(self.checkpoint)

        def get_qy_fillers(speaker, session):
            sdf = exp.fill_df[exp.fill_df["session"] == session]
            return sdf[sdf["speaker"] == speaker]

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
                fillers = get_qy_fillers(speaker, session)

                if len(fillers) == 0:
                    condition["no_filler"] += 1
                    continue

                for ii, da in enumerate(df["da"]):
                    if not valid_qy(da):
                        continue

                    if not valid_qy_post_pause(ii, df):
                        condition["post_pause"] += 1
                        continue

                    if not valid_qy_isolation(ii, df, other_df, min_isolation):
                        condition["isolation"] += 1
                        continue

                    ##################################################
                    # Load audio
                    ##################################################
                    utt = df.iloc[ii]
                    start = utt.end - self.context

                    if start < 0:
                        condition["no_context"] += 1
                        continue

                    waveform, _ = load_waveform(
                        join(
                            self.audio_root,
                            self.relative_audio_path[str(session)] + ".wav",
                        ),
                        start_time=start,
                        end_time=utt.end,
                        sample_rate=self.sample_rate,
                    )
                    waveform = waveform.unsqueeze(0)

                    # # Audio for QY without filler
                    audio = [waveform[0].permute(1, 0)]
                    sil_start_time = [20]
                    for fidx in range(len(fillers)):
                        f = fillers.iloc[fidx]
                        filler_dur = f.end - f.start
                        sil_start_time.append(self.context + filler_dur)
                        # Add filler directly after 'qy'
                        wfill = self.load_filler_audio(f)
                        wfill = torch.cat((waveform, wfill), dim=-1)
                        x_tmp = self.pad_silence(wfill, silence=self.silence)
                        audio.append(x_tmp[0].permute(1, 0))

                    channel = 0 if speaker == "A" else 1
                    sil_start_frames = (
                        (torch.tensor(sil_start_time) * self.frame_hz).round().long()
                    )
                    x = pad_sequence(audio, batch_first=True).permute(0, 2, 1)

                    crosses = []
                    if x.shape[0] < max_batch:
                        out = self.model.probs(x.to(self.model.device))
                        for p, sil_start in zip(
                            out["p_now"][..., channel], sil_start_frames
                        ):
                            cross_idx = self.find_cross(p[sil_start:])
                            crosses.append(cross_idx)
                    else:
                        ii = 0
                        for xx in x.split(max_batch, dim=0):
                            out = self.model.probs(xx.to(self.model.device))
                            n = len(xx)
                            fstarts = sil_start_frames[range(ii, ii + n)]
                            for p, sil_start in zip(
                                out["p_now"][..., channel], fstarts
                            ):
                                cross_idx = self.find_cross(p[sil_start:])
                                crosses.append(cross_idx)
                            ii += n

                    # Check if no-filler was a shift
                    if not valid_qy_by_cross(crosses):
                        condition["no_shift"] += 1
                        continue

                    # Add values
                    questions.append(
                        {
                            "is_filler": 0,
                            "cross": crosses[0],
                            "q_idx": utt.utt_idx,
                            "filler": "qy",
                        }
                    )
                    for i in range(1, len(crosses)):
                        fidx = i - 1
                        f = fillers.iloc[fidx]
                        questions.append(
                            {
                                "is_filler": 1,
                                "filler": f.filler,
                                "cross": crosses[i],
                                "q_idx": utt.utt_idx,
                            }
                        )

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
        filler = df[df["is_filler"] == 1]
        no_filler = df[df["is_filler"] == 0]

        ############################################3
        filler.loc[:, "event"] = True
        filler.loc[filler["cross"] < 0, "event"] = False
        filler.loc[filler["cross"] < 0, "cross"] = max_silence_frames

        ############################################3
        no_filler.loc[:, "event"] = True
        no_filler.loc[no_filler["cross"] < 0, "event"] = False
        no_filler.loc[no_filler["cross"] < 0, "cross"] = max_silence_frames

        # Survival plot
        t_qy, p_qy = kaplan_meier_estimator(
            event=no_filler["event"], time_exit=no_filler["cross"]
        )
        t_qy_filler, p_qy_filler = kaplan_meier_estimator(
            event=filler["event"], time_exit=filler["cross"]
        )
        Result.plot_survival(
            [t_qy, p_qy, t_qy_filler, p_qy_filler], title="QY + Fillers"
        )
        return {
            "time": {"qy": t_qy, "filler": t_qy_filler},
            "probs": {"qy": p_qy, "filler": p_qy_filler},
        }

    @staticmethod
    def plot_survival(TP, chisq=None, pvalue=None, title="Filler and Omission"):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        if chisq is None:
            fig.suptitle(title)
        else:
            fig.suptitle(f"{title}\nChisq={round(chisq, 1)}, P={pvalue}")
        ax.step(TP[2] / 50, TP[3], where="post", color="r", label="Filler")
        ax.step(TP[0] / 50, TP[1], where="post", color="teal", label="Omit")
        ax.set_ylabel("est. probability of shift")
        ax.set_xlabel("time s")
        ax.set_ylim([0, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()


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
        exp = Experiment()
        omission_df = exp.omit_fillers()
        omission_df.to_csv("results/exp1_omission.csv", index=False)
    experiment1_out = Result.survival_omission(omission_df)

    # Experiment 2
    if exists("results/exp2_qy.csv"):
        qy_df = pd.read_csv("results/exp2_qy.csv")
    else:
        exp = Experiment()
        qy_df = exp.experiment_qy(max_batch=5)
        qy_df.to_csv("results/exp2_qy.csv", index=False)

    experiment1_out = Result.survival_qy(qy_df)
