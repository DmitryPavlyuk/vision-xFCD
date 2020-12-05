require(needs)

needs(R1magic)
N <- 100
K <- 4
M <- 50
phi <- GaussianMatrix(N,M)
phi<-matrix(rbinom(n=N*M,prob = 0.1,size = 1),M,N)
xorg <- sparseSignal(N, K, nlev=1e-3)
y <- phi %*% xorg ;# generate measurement
T <- diag(N) ;# Do identity transform
p <- matrix(0, N, 1) ;# initial guess
ll <- solveL1(phi, y, T, p)
x1 <- ll$estimate ;# Returns nlm object
plot(xorg, type="l")
lines(x1, col="red")


needs(imager)
plot(boats)
gc<-resize(boats, 256/8,384/8)
dim(boats)
gc <-grayscale(gc)
plot(gc)
dim(gc)
bdf <- as.data.frame(gc)
head(bdf)
N <- nrow(bdf)
N
M <- 300
phi<-matrix(rbinom(n=N*M,prob = 0.001,size = 1),M,N)
phi <- GaussianMatrix(N,M)
xorg <- bdf$value
y <- phi %*% xorg ;# generate measurement
T <- diag(N) ;# Do identity transform
p <- matrix(0, N, 1) ;# initial guess
ll <- solveL1(phi, y, T, p)
x1 <- ll$estimate ;# Returns nlm object

bdf2<-bdf
bdf2$value<-x1
  
layout(t(1:2))
plot(as.cimg(bdf))
plot(as.cimg(bdf2))
sum(xorg)
sum(x1)
plot(xorg, type="l")
lines(x1, col="red")
