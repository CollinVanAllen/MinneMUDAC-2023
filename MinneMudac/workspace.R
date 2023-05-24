library(tidyverse)
library(lubridate)
library(data.table)
library(readr)
library(DataExplorer)
library(arsenal)
library(mlr)
library(kknn)
library(glmnet)
library(h2o)
library(evtree)
# library(crs)
# library(vroom)
# library(baseballr)

sched23 <- read_csv("MinneMUDAC2023/2023_MLBSchedule.csv")
scheds <- read_csv("MinneMUDAC2023/OriginalSchedules.csv",
                   col_types = cols(DateofMakeup = col_character()),
                   locale = locale(encoding = "UTF-16LE"))
logs <- read_csv("MinneMUDAC2023/GameLogs.csv", 
                 col_types = cols(Completition_Information = col_character(), 
                                  Forfeit_Information = col_character(), 
                                  Protest_Information = col_character(),
                                  LFUmp_ID = col_character(),
                                  RFUmp_ID = col_character(),
                                  Additional_Information = col_character()), 
                 locale = locale(encoding = "UTF-16LE"))


logs$Date <- ymd(logs$Date)
logs$Year <- lubridate::year(logs$Date)

logs_pos <- logs %>% mutate_at(vars(contains("_Position")),
                               funs(case_when(.=="1"~"Pitcher",
                                              .=="2"~"Catcher",
                                              .=="3"~"First",
                                              .=="4"~"Second",
                                              .=="5"~"Third",
                                              .=="6"~"Short",
                                              .=="7"~"Left",
                                              .=="8"~"Center",
                                              .=="9"~"Right")))

logs <- logs %>% mutate(HomeTeam = ifelse(HomeTeam == 'FLO' | HomeTeam == "MIA", "FLO/MIA", HomeTeam))
logs <- logs %>% mutate(VisitingTeam = ifelse(VisitingTeam == 'FLO' | VisitingTeam == "MIA", "FLO/MIA", VisitingTeam))
logs <- logs %>% filter(HomeTeam != "MON" & VisitingTeam != "MON") 


# mnt01, syd01

na_count <- sapply(logs, function(y) sum(length(which(is.na(y)))))

na_count <- data.frame(na_count)

clean_logs <- logs %>% select(-c(Forfeit_Information, Protest_Information, 
                                 LFUmp_ID, RFUmp_ID, Completition_Information, 
                                 Additional_Information, SavingPitcher_ID))


# Removing rest of games for 2001 post 9/11 (Not sure how necessary this portion is)
# Removes 281 entries
clean_logs <- clean_logs %>% filter(!between(Date, '2001-09-15', '2001-12-31'))
# Removing games post Covid pre-June 2021 (Looks like low attendance, then rises in June)
# Removes about 1895 entries
clean_logs <- clean_logs %>% filter(!between(Date, '2020-03-15', '2021-06-15'))
# Removing games with 0 attendance?
# Removes 1180 entries
clean_logs <- clean_logs %>% filter(Attendance > 0)


mn_logs <- clean_logs %>% filter(HomeTeam == "MIN")


# Graphs --------------------------------------------------------------------

clean_logs %>% filter(HomeTeam == "MIN") %>% ggplot(aes(y = Attendance,x=Date)) +
  geom_point() + geom_line() + facet_wrap(~DayNight)

clean_logs %>% 
  ggplot(aes(y = Attendance, x = Year, group = Year)) +
  geom_boxplot() + 
  geom_smooth(se=FALSE, color="red", aes(group=1)) +
  facet_wrap(~HomeTeam)

clean_logs %>% filter(HomeTeam == "MIN") %>%  ggplot(aes(Attendance, fill = DayNight)) +
  geom_histogram() + facet_wrap(~DayofWeek + DayNight)

logs %>%  ggplot(aes(Attendance, fill = DayNight)) +
  geom_histogram() + facet_wrap(~DayofWeek + DayNight)


test <- clean_logs %>% filter(lubridate::year(Date) == 2022 & DayNight == "D")
#-------------------------------------------------------------------------
ml_logs <- clean_logs[,c(1,3:9,13,15)]
ml_logs <- mutate_if(ml_logs, is.character, as.factor)
ml_logs <- ml_logs %>% mutate(Month = lubridate::month(Date), Year = lubridate::year(Date))
ml_logs <- ml_logs %>% drop_columns(1)
# logsSlim[c(1,2,4,6)] <- lapply(logsSlim[c(1,2,4,6)], factor)
ml_logs <- ml_logs %>% drop_na()

attendanceTask <- makeRegrTask(data = ml_logs, target = "Attendance")

# GLMNET-------------------------------------------------------------
glmnet <- makeLearner("regr.glmnet") 
glmnetModel <- train(glmnet, attendanceTask)
glmnetPred <- predict(glmnetModel, newdata = ml_logs)

performance(glmnetPred, measures = list(rsq, mape, expvar))


kFold <- makeResampleDesc(method = "CV", iters = 25,
                          stratify = FALSE)
kFoldCV <- resample(learner = glmnet, task = attendanceTask,
                    resampling = kFold, measures = list(rsq,mape,expvar))
kFoldCV$aggr

# GBM-------------------------------------------------------------
gbm <- makeLearner("regr.h2o.gbm") 
gbmModel <- train(gbm, attendanceTask)
gbmPred <- predict(gbmModel, newdata = ml_logs)

performance(gbmPred, measures = list(rsq,mape,expvar))


kFold <- makeResampleDesc(method = "CV", iters = 25,
                          stratify = FALSE)
kFoldCV <- resample(learner = gbm, task = attendanceTask,
                    resampling = kFold, measures = list(rsq,mape,expvar))
kFoldCV$aggr

# knn
knn <- makeLearner("regr.kknn", k = 20) 
knnModel <- train(knn, attendanceTask)
knnPred <- predict(knnModel, newdata = ml_logs)

performance(knnPred, measures = list(rsq,mape,expvar))


kFold2 <- makeResampleDesc(method = "CV", iters = 25,
                          stratify = FALSE)
kFoldCV <- resample(learner = knn, task = attendanceTask,
                    resampling = kFold2, measures = list(rsq,mape,expvar))
kFoldCV$aggr

# forest
forest <- makeLearner("regr.h2o.randomForest") 
forestModel <- train(forest, attendanceTask)
forestPred <- predict(forestModel, newdata = ml_logs)

performance(forestPred, measures = list(rsq,mape,expvar))


kFold3 <- makeResampleDesc(method = "CV", iters = 25,
                           stratify = FALSE)
kFoldCV3 <- resample(learner = forest, task = attendanceTask,
                    resampling = kFold3, measures = list(rsq,mape,expvar))
kFoldCV3$aggr
