
?read.csv
mtcars <- read.csv('mtcars.csv', row.names = c(1))

fuelEconomy <- lm(mpg ~ cyl + wt, data = mtcars)
summary(fuelEconomy)

par(mfrow = c(2,2))
plot(fuelEconomy)
par(mfrow = c(1,1))