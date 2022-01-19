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
dat_server = read.table("data/centralized_run.txt",sep=",",quote="\'",header=F)
names(dat_server) <- c("run", "loss", "accuracy", "correct", "total")
dat_server <- mutate_at(dat_server, vars("run"), as.factor)
dat_server[dat_server$run=="test",]$loss <- dat_server[dat_server$run=="test",]$loss/(dat_server[dat_server$run=="test",]$total/100)
dat_server[dat_server$run=="train",]$loss <- dat_server[dat_server$run=="train",]$loss/(dat_server[dat_server$run=="train",]$total/128)
dat_server <- cbind(epoch=rep(1:(length(dat_server$loss)/2), each=2), dat_server)
str(dat_server)

pserver <- ggplot(dat_server, aes(x=epoch, y=loss, color=run))+
              labs(title = "Server Training",
                  subtitle = "CIFAR, full 50000/10000 data",
                  caption = "Jan, 2022",
                  tag = "Figure 1",
                  x = "Epoch",
                  y = "Loss",
                  color = "Run"
               ) +
            geom_line() + theme_classic()
pserver

fl_cmp_train <- select(dat_server[dat_server$run=="train",], epoch, server=loss)
fl_cmp_test <- select(dat_server[dat_server$run=="test",], epoch, server=loss)

# single-node-subset-based training
#
dat_sin = read.table("data/single.txt",sep=",",quote="\'",header=F)
names(dat_sin) <- c("run", "loss", "accuracy", "correct", "total")
dat_sin <- mutate_at(dat_sin, vars("run"), as.factor)
dat_sin[dat_sin$run=="test",]$loss <- dat_sin[dat_sin$run=="test",]$loss/(dat_sin[dat_sin$run=="test",]$total/100)
dat_sin[dat_sin$run=="train",]$loss <- dat_sin[dat_sin$run=="train",]$loss/(dat_sin[dat_sin$run=="train",]$total/128)
dat_sin <- cbind(epoch=rep(1:(length(dat_sin$loss)/2), each=2), dat_sin)
str(dat_sin)

psingle <- ggplot(dat_sin, aes(x=epoch, y=loss, color=run))+
  labs(title = "Single node, 1/3 dataset training",
       subtitle = "CIFAR, 1/3 of full data chunk ",
       caption = "Jan, 2022",
       tag = "Figure 2",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) +
  geom_line() + theme_classic()
psingle

fl_cmp_train <- cbind(fl_cmp_train, select(dat_sin[dat_sin$run=="train",], single=loss))
fl_cmp_test <- cbind(fl_cmp_test, select(dat_sin[dat_sin$run=="test",], single=loss))

# single-node-federated training
#
dat_fed = read.table("data/loss_acc_tracking.txt",sep=",",quote="\'",header=F)
names(dat_fed) <- c("run", "loss", "accuracy", "correct", "total")
dat_fed <- mutate_at(dat_fed, vars("run"), as.factor)
dat_fed[dat_fed$run=="test",]$loss <- dat_fed[dat_fed$run=="test",]$loss/(dat_fed[dat_fed$run=="test",]$total/100)
dat_fed[dat_fed$run=="train",]$loss <- dat_fed[dat_fed$run=="train",]$loss/(dat_fed[dat_fed$run=="train",]$total/128)
dat_fed <- cbind(epoch=rep(1:(length(dat_fed$loss)/2), each=2), dat_fed)
str(dat_fed)

pfed <- ggplot(dat_fed, aes(x=epoch, y=loss, color=run))+
  labs(title = "Single node, Federated training",
       subtitle = "CIFAR, Federated training with local 1/3 data chunk",
       caption = "Jan, 2022",
       tag = "Figure 3",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) +
  geom_line() + theme_classic()
pfed

fl_cmp_train <- merge.data.frame(fl_cmp_train, select(dat_fed[dat_fed$run=="train",], epoch, federated=loss), all=TRUE)
fl_cmp_test <- merge.data.frame(fl_cmp_test, select(dat_fed[dat_fed$run=="test",], epoch, federated=loss), all=TRUE)


str(fl_cmp_train)
flcmp_m <- melt(fl_cmp_train, id=c("epoch"))

pcmp1 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Server, Single and Federated training, Train loss",
       subtitle = "CIFAR (single/federated training with local 1/3 data chunk)",
       caption = "Jan, 2022",
       tag = "Figure 4",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) + scale_y_continuous(limits = c(0, 2.1)) +
  geom_line() + theme_classic()
pcmp1

str(fl_cmp_test)
flcmp_m <- melt(fl_cmp_test, id=c("epoch"))

pcmp2 <- ggplot(flcmp_m, aes(x=epoch, y=value, color=variable))+
  labs(title = "Server, Single and Federated training, Test loss",
       subtitle = "CIFAR (single/federated training with local 1/3 data chunk)",
       caption = "Jan, 2022",
       tag = "Figure 5",
       x = "Epoch",
       y = "Loss",
       color = "Run"
  ) + scale_y_continuous(limits = c(0, 2.1)) +
  geom_line() + theme_classic()
pcmp2

grid.arrange(pcmp1, pcmp2, ncol=2)
