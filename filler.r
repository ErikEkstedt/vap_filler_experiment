library(lsmeans)
library(tidyverse)
library(moderndive)
library(infer)
library(lme4)
library(reshape2)
library(ggpubr)
library(arm)
library(dplyr)
library(MuMIn)
library(plyr)
library(ggplot2)
library(survival)
library(ranger)
library(ggfortify)
library(survminer)
#library(lmerTest)
rescale <- arm::rescale

#https://rviews.rstudio.com/2017/09/25/survival-analysis-with-r/

setwd("~/Desktop/KTH/filler")

uh <- read.csv('uh_full_correct.csv') 
um <- read.csv('um_full_correct.csv')

uh$type <- 'uh'
um$type <- 'um'
df <- rbind(uh, um)
df$fill_dur <- df$fill_end - df$fill_start
df$w_filler <- as.factor(df$w_filler)

## treating censored as silence = 10s
df.surv <- df %>%
  mutate(
    p_now = replace(p_now, p_now == -100, 999),
    p_fut = replace(p_fut, p_fut == -100, 999),
    status.pnow = ifelse(p_now < 500, 1, 0),
    status.pfut = ifelse(p_fut < 500, 1, 0),
    w_filler = as.factor(w_filler),
    type = as.factor(type),
    fill_dur.rs = rescale(fill_dur),
    w_filler.rs = rescale(w_filler),
    p_now.rs = rescale(p_now),
    p_fut.rs = rescale(p_fut),
    type.rs = rescale(type),
    status.pnow.rs = rescale(status.pnow),
    status.pfut.rs = rescale(status.pfut)
  )
## Q: we probably need to rescale, but why there is a line...?

## models for examining with/without fillers
km <- with(df.surv, Surv(p_now, status.pnow))
head(km, 80)
survdiff(km ~ w_filler, data = df.surv)


km_fit <- survfit(Surv(p_now, status.pnow) ~ 1, data=df.surv)
summary(km_fit)
autoplot(km_fit)

# no need to rescale, only one factor
km_fit_filler <- survfit(Surv(p_now, status.pnow) ~ w_filler, data=df.surv)
summary(km_fit_filler)
autoplot(km_fit_filler)+theme_bw()
ggsurvplot(km_fit_filler, conf.int = 'True')

cox <- coxph(Surv(p_now, status.pnow) ~ w_filler.rs, data = df.surv)
summary(cox)
cox_fit <- survfit(cox)
autoplot(cox_fit)

aa_fit <- aareg(Surv(p_now, status.pnow) ~ w_filler + type, data = df.surv)
aa_fit
autoplot(aa_fit)

#######################
## with filler models

df.filler <- subset(df.surv, w_filler==1)
prosody <- read.table('filler_output.txt',sep=' ', header = TRUE)
prosody <- subset(prosody, WITH_OR_OMIT_FILLER==1)
df.filler$f0 <- prosody$FILLER_F0
df.filler$intensity <- prosody$FILLER_INT

## with filler, type and duration effects
cox.f <- coxph(Surv(p_now, status.pnow) ~ fill_dur.rs + type.rs, data = df.filler)
summary(cox.f)
cox.f.fit <-survfit(cox.f)
autoplot(cox.f.fit)

aa.f <- aareg(Surv(p_now, status.pnow) ~ fill_dur.rs + type.rs, data = df.filler)
aa.f
autoplot(aa.f)

# type, dur, f0, intensity
cox.prosody <- coxph(Surv(p_now, status.pnow) ~ fill_dur.rs + type.rs + f0 + intensity, data = df.filler)
# if doesn't rescale prosody, doesn't fit

cox.prosody2 <- coxph(Surv(p_now, status.pnow) ~ type.rs * (fill_dur.rs  + f0.rs + intensity.rs), data = df.filler)
cox.prosody <- coxph(Surv(p_now, status.pnow) ~ type.rs + fill_dur.rs  + f0.rs + intensity.rs , data = df.filler)
anova(cox.prosody, cox.prosody2, test = 'Chisq')
# not significant, interaction doesn't matter
summary(cox.prosody)

cox.prosody3 <- coxph(Surv(p_now, status.pnow) ~ type.rs + fill_dur.rs  + f0.rs * intensity.rs , data = df.filler)
summary(cox.prosody3) # interaction again not significant
anova(cox.prosody, cox.prosody3, test = 'Chisq') # not significant

#so, we use cox.prosody


aa.prosody <- aareg(Surv(p_now, status.pnow) ~ fill_dur.rs + type.rs+ f0 + intensity, data = df.filler)

df.filler$f0.rs <- rescale(df.filler$f0)
df.filler$intensity.rs <- rescale(df.filler$intensity)

aa.prosody <- aareg(Surv(p_now, status.pnow) ~ fill_dur.rs + type.rs+ f0.rs + intensity.rs , data = df.filler)
aa.prosody

####################### PAST ##################
## USELESS ##
## below: previous lme models
df_rs <- df %>%
  mutate(
    fil_dur.rs = rescale(fil_dur),
    w_filler.rs = rescale(w_filler),
    p_now.rs = rescale(p_now),
    p_fut.rs = rescale(p_fut),
    type.rs = rescale(type)
  )



uh_sub <-subset(uh, p_now!=-100 & p_fut!=-100)
uh_sub$withfiller <-as.factor(uh_sub$withfiller)
ggplot(uh_sub, aes(x = fil_dur, y = p_now, group = withfiller))+geom_point(aes(color=withfiller))+geom_smooth(aes(x=fil_dur,y=p_now,color=withfiller))

ggplot(uh_sub, aes(x = fil_dur, y = p_now, group = withfiller))+geom_point(aes(color=withfiller))+facet_grid(~session)

ggplot(uh_sub, aes(x = fil_dur, y = p_fut, group = withfiller))+geom_point(aes(color=withfiller))

ggplot(uh_sub, aes(x = withfiller, y=p_now))+geom_boxplot()

mod <- lmer(p_now ~ w_filler.rs *type.rs + (1|session), data=df_rs)
summary(mod) #ref p.266

mod_lmertest <- lmerTest::lmer()
