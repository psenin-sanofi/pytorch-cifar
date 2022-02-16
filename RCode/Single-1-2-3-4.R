require(ggplot2)
require(reshape)
require(dplyr)
require(scales)
require(graphics)
require(grid)
require(gridExtra)
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan",
                                 "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

# server-based training
#
dat_1 = read.table("data/centralized_run.txt",sep=",",quote="\'",header=F)
names(dat_1) <- c("run", "loss", "accuracy", "correct", "total")
dat_1 <- mutate_at(dat_1, vars("run"), as.factor)
dat_1[dat_1$run=="test",]$loss <- dat_1[dat_1$run=="test",]$loss/(dat_1[dat_1$run=="test",]$total/100)
dat_1[dat_1$run=="train",]$loss <- dat_1[dat_1$run=="train",]$loss/(dat_1[dat_1$run=="train",]$total/128)
dat_1 <- cbind(epoch=rep(1:(length(dat_1$loss)/2), each=2), dat_1)
str(dat_1)

fl_cmp_1_train <- select(dat_1[dat_1$run=="train",], epoch, full=loss)
fl_cmp_1_test <- select(dat_1[dat_1$run=="test",], epoch, full=loss)


# split 2
#
dat_2 = read.table("data/subset_2.txt",sep=",",quote="\'",header=F)
names(dat_2) <- c("run", "loss", "accuracy", "correct", "total")
dat_2 <- mutate_at(dat_2, vars("run"), as.factor)
dat_2[dat_2$run=="test",]$loss <- dat_2[dat_2$run=="test",]$loss/(dat_2[dat_2$run=="test",]$total/100)
dat_2[dat_2$run=="train",]$loss <- dat_2[dat_2$run=="train",]$loss/(dat_2[dat_2$run=="train",]$total/128)
dat_2 <- cbind(epoch=rep(1:(length(dat_2$loss)/2), each=2), dat_2)

fl_cmp_2_train <- select(dat_2[dat_2$run=="train",], epoch, split_2=loss)
fl_cmp_2_test <- select(dat_2[dat_2$run=="test",], epoch, split_2=loss)

# split 3
#
dat_3 = read.table("data/subset_3.txt",sep=",",quote="\'",header=F)
names(dat_3) <- c("run", "loss", "accuracy", "correct", "total")
dat_3 <- mutate_at(dat_3, vars("run"), as.factor)
dat_3[dat_3$run=="test",]$loss <- dat_3[dat_3$run=="test",]$loss/(dat_3[dat_3$run=="test",]$total/100)
dat_3[dat_3$run=="train",]$loss <- dat_3[dat_3$run=="train",]$loss/(dat_3[dat_3$run=="train",]$total/128)
dat_3 <- cbind(epoch=rep(1:(length(dat_3$loss)/2), each=2), dat_3)

fl_cmp_3_train <- select(dat_3[dat_3$run=="train",], epoch, split_3=loss)
fl_cmp_3_test <- select(dat_3[dat_3$run=="test",], epoch, split_3=loss)

# split 4
#
dat_4 = read.table("data/subset_4.txt",sep=",",quote="\'",header=F)
names(dat_4) <- c("run", "loss", "accuracy", "correct", "total")
dat_4 <- mutate_at(dat_4, vars("run"), as.factor)
dat_4[dat_4$run=="test",]$loss <- dat_4[dat_4$run=="test",]$loss/(dat_4[dat_4$run=="test",]$total/100)
dat_4[dat_4$run=="train",]$loss <- dat_4[dat_4$run=="train",]$loss/(dat_4[dat_4$run=="train",]$total/128)
dat_4 <- cbind(epoch=rep(1:(length(dat_4$loss)/2), each=2), dat_4)

fl_cmp_4_train <- select(dat_4[dat_4$run=="train",], epoch, split_4=loss)
fl_cmp_4_test <- select(dat_4[dat_4$run=="test",], epoch, split_4=loss)


# merge and plot
#
fl_cmp_train <- merge.data.frame(merge.data.frame(fl_cmp_1_train, fl_cmp_2_train, all=TRUE), 
                                merge.data.frame(fl_cmp_3_train, fl_cmp_4_train, all=TRUE), all=TRUE)
fl_cmp_test <- merge.data.frame(merge.data.frame(fl_cmp_1_test, fl_cmp_2_test, all=TRUE), 
                                merge.data.frame(fl_cmp_3_test, fl_cmp_4_test, all=TRUE), all=TRUE)


str(fl_cmp_train)
flcmp_m <- melt(fl_cmp_train, id=c("epoch"))

pcmp1 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Full and split subset training, train loss",
       subtitle = "CIFAR data",
       caption = "Jan, 2022",
       tag = "Figure 6",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) + scale_y_continuous(limits = c(0, 2.1)) +
  geom_line() + theme_classic()
pcmp1

str(fl_cmp_test)
flcmp_m <- melt(fl_cmp_test, id=c("epoch"))

pcmp2 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Full and split subset training, test loss",
       subtitle = "CIFAR data",
       caption = "Jan, 2022",
       tag = "Figure 7",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) + scale_y_continuous(limits = c(0, 2.1)) +
  geom_line() + theme_classic()
pcmp2

grid.arrange(pcmp1, pcmp2, ncol=2, 
             top=textGrob("Train and test loss for a range of splits",
                          gp=gpar(fontsize=20,font=2)))


# server-based training
#
dat_1 = read.table("data/centralized_run.txt",sep=",",quote="\'",header=F)
names(dat_1) <- c("run", "loss", "accuracy", "correct", "total")
dat_1 <- mutate_at(dat_1, vars("run"), as.factor)
dat_1[dat_1$run=="test",]$loss <- dat_1[dat_1$run=="test",]$loss/(dat_1[dat_1$run=="test",]$total/100)
dat_1[dat_1$run=="train",]$loss <- dat_1[dat_1$run=="train",]$loss/(dat_1[dat_1$run=="train",]$total/128)
dat_1 <- cbind(epoch=rep(1:(length(dat_1$loss)/2), each=2), dat_1)
str(dat_1)

fl_cmp_1_train <- select(dat_1[dat_1$run=="train",], epoch, full=accuracy)
fl_cmp_1_test <- select(dat_1[dat_1$run=="test",], epoch, full=accuracy)


# split 2
#
dat_2 = read.table("data/subset_2.txt",sep=",",quote="\'",header=F)
names(dat_2) <- c("run", "loss", "accuracy", "correct", "total")
dat_2 <- mutate_at(dat_2, vars("run"), as.factor)
dat_2[dat_2$run=="test",]$loss <- dat_2[dat_2$run=="test",]$loss/(dat_2[dat_2$run=="test",]$total/100)
dat_2[dat_2$run=="train",]$loss <- dat_2[dat_2$run=="train",]$loss/(dat_2[dat_2$run=="train",]$total/128)
dat_2 <- cbind(epoch=rep(1:(length(dat_2$loss)/2), each=2), dat_2)

fl_cmp_2_train <- select(dat_2[dat_2$run=="train",], epoch, split_2=accuracy)
fl_cmp_2_test <- select(dat_2[dat_2$run=="test",], epoch, split_2=accuracy)

# split 3
#
dat_3 = read.table("data/subset_3.txt",sep=",",quote="\'",header=F)
names(dat_3) <- c("run", "loss", "accuracy", "correct", "total")
dat_3 <- mutate_at(dat_3, vars("run"), as.factor)
dat_3[dat_3$run=="test",]$loss <- dat_3[dat_3$run=="test",]$loss/(dat_3[dat_3$run=="test",]$total/100)
dat_3[dat_3$run=="train",]$loss <- dat_3[dat_3$run=="train",]$loss/(dat_3[dat_3$run=="train",]$total/128)
dat_3 <- cbind(epoch=rep(1:(length(dat_3$loss)/2), each=2), dat_3)

fl_cmp_3_train <- select(dat_3[dat_3$run=="train",], epoch, split_3=accuracy)
fl_cmp_3_test <- select(dat_3[dat_3$run=="test",], epoch, split_3=accuracy)

# split 4
#
dat_4 = read.table("data/subset_4.txt",sep=",",quote="\'",header=F)
names(dat_4) <- c("run", "loss", "accuracy", "correct", "total")
dat_4 <- mutate_at(dat_4, vars("run"), as.factor)
dat_4[dat_4$run=="test",]$loss <- dat_4[dat_4$run=="test",]$loss/(dat_4[dat_4$run=="test",]$total/100)
dat_4[dat_4$run=="train",]$loss <- dat_4[dat_4$run=="train",]$loss/(dat_4[dat_4$run=="train",]$total/128)
dat_4 <- cbind(epoch=rep(1:(length(dat_4$loss)/2), each=2), dat_4)

fl_cmp_4_train <- select(dat_4[dat_4$run=="train",], epoch, split_4=accuracy)
fl_cmp_4_test <- select(dat_4[dat_4$run=="test",], epoch, split_4=accuracy)


# merge and plot
#
fl_cmp_train <- merge.data.frame(merge.data.frame(fl_cmp_1_train, fl_cmp_2_train, all=TRUE), 
                                 merge.data.frame(fl_cmp_3_train, fl_cmp_4_train, all=TRUE), all=TRUE)
fl_cmp_test <- merge.data.frame(merge.data.frame(fl_cmp_1_test, fl_cmp_2_test, all=TRUE), 
                                merge.data.frame(fl_cmp_3_test, fl_cmp_4_test, all=TRUE), all=TRUE)


str(fl_cmp_train)
flcmp_m <- melt(fl_cmp_train, id=c("epoch"))

pcmp1 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Full and split subset training, train accuracy",
       subtitle = "CIFAR data",
       caption = "Jan, 2022",
       tag = "Figure 8",
       x = "Epoch",
       y = "Accuracy",
       color = "Run"
  ) + scale_y_continuous(limits = c(20, 100),
                         minor_breaks = seq(20 , 100, 5), 
                         breaks = seq(20, 100, 10)) +
  scale_x_continuous(limits = c(0, 250)) +
  geom_line() + theme_classic() +
  theme(panel.grid.major = element_line(colour="grey", size=0.3, linetype = "dotted"))
pcmp1

str(fl_cmp_test)
flcmp_m <- melt(fl_cmp_test, id=c("epoch"))

pcmp2 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Full and split subset training, test accuracy",
       subtitle = "CIFAR data",
       caption = "Jan, 2022",
       tag = "Figure 9",
       x = "Epoch",
       y = "Accuracy",
       color = "Run"
  ) + scale_y_continuous(limits = c(50, 100),
                         minor_breaks = seq(20 , 100, 5), 
                         breaks = seq(20, 100, 10)) +
  scale_x_continuous(limits = c(0, 250)) +
  geom_line() + theme_classic() +
  theme(panel.grid.major = element_line(colour="grey", size=0.3, linetype = "dotted"))
pcmp2

grid.arrange(pcmp1, pcmp2, ncol=2, 
             top=textGrob("Train and test accuracy for a range of splits",
                          gp=gpar(fontsize=20,font=2)))
