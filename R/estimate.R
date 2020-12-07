


TGMCS <- function(Q,Lw, omega,accMask=omega, mu=0.0001, lambda1=0.01, lambda2=0.05, lambda3=0.05, r=1, maxIter=1000, tol=1e-4, returnQhat=F){
  T<-ncol(Q)
  N<-nrow(Q)
  H <- matrix(0,T,T-1)
  for (i in 1:(T-1)){
    H[i,i]<-(-1)
    H[i+1,i]<-1
  }
  U <- matrix(rnorm(N*r),N,r)
  E <- matrix(rnorm(T*r),T,r)
  i <- 1
  Qhat<-U%*%t(E)
  prevObsMAE <- sum(hadamard.prod(omega,abs(Q-Qhat)))/sum(omega>0)
  converged <- FALSE
  while(i<maxIter){
    FR<-hadamard.prod(omega, U%*%t(E)-Q)
    U <- U - mu*(FR%*%E+2*lambda1*U+lambda2*(Lw+t(Lw))%*%U)
    E <- E - mu*(t(FR)%*%U + 2*lambda1*E+2*lambda3*H%*%t(H)%*%E)
    Qhat<-U%*%t(E)
    curObsMAE <- sum(hadamard.prod(omega,abs(Q-Qhat)))/sum(omega>0)
    #print(curObsMAE)
    if (abs(curObsMAE-prevObsMAE)<tol){
      converged <- TRUE
      break;
    }else{
      prevObsMAE<-curObsMAE
    }
    i <- i + 1
    if (i==maxIter){
      warning("No convergence - maximum number of iterations reached")
      converged <- FALSE
    } 
  }
  unobsaccMask <- accMask+1
  unobsaccMask[unobsaccMask==2]<-0
  print(paste("converged = ",converged))
  res<-(list(observedMAE=sum(hadamard.prod(accMask,abs(Q-Qhat)))/sum(accMask>0),
              unobservedMAE=sum(hadamard.prod(unobsaccMask,abs(Q-Qhat)))/sum(unobsaccMask>0),
              observedMAPE=sum(hadamard.prod(accMask,abs(Q-Qhat)/(Q+0.0001)))/sum(accMask>0),
              unobservedMAPE=sum(hadamard.prod(unobsaccMask,abs(Q-Qhat)/(Q+0.0001)))/sum(unobsaccMask>0),
              converged=converged))
  if(returnQhat) res[['Qhat']]=U%*%t(E)
  return(res)
}
extendObs<-function(omega, observable){
  omegaExt <- omega
  for (t in 1:ncol(omega)){
    for (i in 1:nrow(omega)){
      if (omega[i,t]>0){#observed
        name<-names(omega[i,t])
        obs<-observable[[name]]
        omegaExt[obs,t]<-1
      }
    }
  }
  return(omegaExt)
}

randomObs<-function(N,T, sparsity, Q){
  obs<-matrix(0,N,T)
  rownames(obs)<-rownames(Q)
  for (t in 1:T){
    for (n in 1:N){
      s<-rbinom(1, size = round(Q[n,t]), prob = sparsity)
      #o<-sample(rownames(obs), size=s, prob=qs, replace = TRUE)
      obs[n,t]<-(s>0)
    }
  }
  return(obs)
}




meta<-read.csv("X:\\Dmitry\\projects\\EWGT2021\\R\\d07_text_meta_2020_11_16.txt", header=T, sep="\t")
stations<-c(717046, 717045,
            717263, 717264,
            716943, 716942,
            716331, 717445,
            717047, 716028,
            716946, 718085,
            718173,
            716939)

observable<-list('717046'='717045', '717045'='717046',
                 '717263'='717264', '717264'='717263',
                 '716943'='716942', '716942'='716943',
                 '716331'='717445', '717445'='716331',
                 '717047'='716028', '716028'='717047',
                 '716946'='718085', '718085'='716946',
                 '718173'='716939', '716939'='718173'
)

data<-read.csv("X:\\Dmitry\\projects\\EWGT2021\\R\\d07_text_station_5min_2020_11_29.txt", header=F)
data.tb<-as_tibble(data)
data.tb<-data.tb%>%mutate(datetime=as.POSIXct(V1, format="%m/%d/%Y %H:%M:%S"), station=V2, volume=V10, occupancy=V11, speed=V12)%>%
  select(datetime,station,volume, occupancy,speed)

data.tbf<-data.tb%>%filter(station %in% stations)%>%mutate(speed=ifelse(is.na(speed),65,speed))

probs<-data.tbf%>%group_by(station)%>%summarise(mean_flow=mean(volume, na.rm=T))%>%pull(mean_flow)
names(probs)<-stations
sort(stations)
sort(unique(data.tbf$station))

vols<-data.tbf%>%select(datetime, station, volume)%>%mutate(volume=ifelse(is.na(volume),0,volume))%>%pivot_wider(names_from=station, values_from=volume)
rels<-cor(vols%>%select(-datetime))
rels
md<-rowSums(abs(rels))
Lw<--abs(rels)+diag(ncol(rels))+diag(md)

dat<-data.tbf%>%select(datetime, station, speed)%>%pivot_wider(names_from=station, values_from=speed)%>%select(-datetime)
cor(dat)
N <- ncol(dat)
T <- nrow(dat)
Q <- t(dat)

data.tbf%>%select(datetime, station, speed)%>%ggplot(aes(x = datetime, y = speed, col=station, group=station)) + geom_line(size=1)
data.tbf%>%group_by(station)%>%summarise(sd(speed))

sp<-9e-4
omega<-randomObs(N, T, sp, Q)
sum(omega>0)/(N*T)
res<-TGMCS(Q, Lw, omega,accMask=omega, maxIter=1e6, tol=1e-6, r=2,mu=0.0001,lambda1=0.01, lambda2=10, lambda3=10)

omegax2 <-randomObs(N, T, 2*sp, Q)
sum(omegax2>0)/(N*T)
res<-TGMCS(Q, Lw, omegax2,accMask=omega, maxIter=1e6, tol=1e-6, r=2,mu=0.0001,lambda1=0.01, lambda2=10, lambda3=10)
omegaExt <-extendObs(om, observable)
sum(omegaExt>0)/(N*T)
TGMCS(Q, Lw, omegaExt,accMask=omega, maxIter=1e6, tol=1e-6, r=2,mu=0.0001,lambda1=0.01, lambda2=10, lambda3=10)


TGMCS(Q, Lw, matrix(1,N,T),accMask=omega, maxIter=1e6, tol=1e-6, r=2,mu=0.0001,lambda1=0.01, lambda2=10, lambda3=10)


sp<-9e-4
omega<-randomObs(N, T, sp, Q)
sum(omega>0)/(N*T)
res<-TGMCS(Q, Lw, omega,accMask=omega, maxIter=1e3, tol=1e-3, r=15,mu=0.0001,lambda1=0.01, lambda2=0.05, lambda3=0.05, returnQhat=T)
hat.tbf<-as_tibble(t(res$Qhat))
hat.tbf$datetime<-unique(data.tbf%>%pull(datetime))
hat.tbf%>%pivot_longer(-one_of("datetime"), names_to="station", values_to = "speed")%>%ggplot(aes(x = datetime, y = speed, col=station, group=station)) + geom_line(size=1)

res$observedMAE
res$unobservedMAE

data.tbf%>%select(datetime, station, speed)%>%ggplot(aes(x = datetime, y = speed, col=station, group=station)) + geom_line(size=1)
mean(abs(Q-res$Qhat)/Q)

est<-list()
r<-15
mu=0.0001
lambda1=0.01
lambda2=0.05
lambda3=0.05
tol=1e-6
maxIter=1e6
#1e-3,2e-3,3e-3,4e-3,5e-3,9e-3,11e-3
for(rep in 1:50){
  for (sp in c(5e-4)){
    omega <- randomObs(N, T, sp, Q)
    omegax2 <-omega+randomObs(N, T, sp, Q)
    omegax2[omegax2>1]<-1
    omegaExt <-extendObs(omega, observable)
    est[[length(est)+1]]<-c(TGMCS(Q, Lw, omega,accMask=omega,maxIter=maxIter,tol=tol, r=r,mu=mu,lambda1=lambda1, lambda2=lambda2, lambda3=lambda3),
                            sparsity=sp,obslinks=sum(omega>0),coverage=sum(omega>0)/(N*T), name="omega")
    est[[length(est)+1]]<-c(TGMCS(Q, Lw, omegax2,accMask=omegax2,maxIter=maxIter,tol=tol, r=r,mu=mu,lambda1=lambda1, lambda2=lambda2, lambda3=lambda3),
                            sparsity=sp,obslinks=sum(omegax2>0),coverage=sum(omegax2>0)/(N*T), name="omegax2")
    est[[length(est)+1]]<-c(TGMCS(Q, Lw, omegaExt,accMask=omegaExt,maxIter=maxIter,tol=tol, r=r,mu=mu,lambda1=lambda1, lambda2=lambda2, lambda3=lambda3), 
                            sparsity=sp,obslinks=sum(omegaExt>0),coverage=sum(omegaExt>0)/(N*T),name="omegaExt")
    print(tail(bind_rows(est)))
    print(paste(rep,sp))
  }
}
spars<-c()
for (sp in c(5e-4,1e-3,2e-3,3e-3,4e-3,5e-3,9e-3,11e-3)){
  omega<-randomObs(N, T, sp, Q)
  spars[as.character(sp)]<-sum(omega>0)/(N*T)
}
spars


saveRDS(est, file="es.Rds")
est.df<-bind_rows(est)
est.df
f<-1
est.df%>%filter(converged==TRUE)%>%select(name, obslinks, sparsity,unobservedMAE)%>%
  group_by(name, sparsity)%>%
  summarise(meanMAE=mean(unobservedMAE), sdMAE = sd(unobservedMAE), n=n(), minMAE=min(unobservedMAE), maxMAE=max(unobservedMAE),
            lb=max(meanMAE-f*1.96*sdMAE/sqrt(n),minMAE),ub=min(meanMAE+f*1.96*sdMAE/sqrt(n),maxMAE))%>%
  ggplot(aes(x = sparsity, y = meanMAE, col=name, group=name,linetype=name)) + geom_line(size=1)+
  geom_ribbon(aes(ymin=lb, ymax=ub, col=name, group=name,linetype=name), alpha=0.1)

est.df%>%filter(converged==TRUE)%>%select(name, obslinks, sparsity,unobservedMAPE)%>%
  group_by(name, sparsity)%>%
  summarise(meanMAPE=mean(unobservedMAPE), sdMAPE = sd(unobservedMAPE), n=n(), minMAPE=min(unobservedMAPE), maxMAPE=max(unobservedMAPE),
            lb=max(meanMAPE-f*1.96*sdMAPE/sqrt(n),minMAPE),ub=min(meanMAPE+f*1.96*sdMAPE/sqrt(n),maxMAPE))%>%mutate(coverage = spars[as.character(sparsity)])%>%
  ggplot(aes(x = coverage, y = meanMAPE, col=name, group=name,linetype=name)) + geom_line(size=1)+
  geom_ribbon(aes(ymin=lb, ymax=ub, col=name, group=name,linetype=name), alpha=0.1)

est.df%>%filter(converged==TRUE)%>%select(name, obslinks, sparsity,unobservedMAPE)%>%
  group_by(name, sparsity)%>%
  summarise(meanMAPE=mean(unobservedMAPE), meanSp = mean(obslinks)/(N*T))%>%mutate(coverage = round(spars[as.character(sparsity)],2))%>%
  pivot_wider(id_cols=c(name), names_from="coverage", values_from="meanMAPE")


