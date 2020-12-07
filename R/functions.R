
# Implementaion of the temporal geometric matrix completion approach with spatial data
# Zhang, Z., Li, M., Lin, X., Wang, Y., 2020. Network-wide traffic flow estimation with insufficient volume detection and crowdsourcing data. Transportation Research Part C: Emerging Technologies 121, 102870. https://doi.org/10.1016/j.trc.2020.102870
TGMCS <- function(Q,Lw, omega,accMask=omega, mu=1e-4, lambda1=1e-2, lambda2=5e-2, lambda3=5e-2, r=nrow(Q), maxIter=1e6, tol=1e-6, returnQhat=F){
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
  prevObsMAE <- omegaMAE(omega,Q,Qhat)
  converged <- FALSE
  while(i<maxIter){
    FR<-hadamard.prod(omega, U%*%t(E)-Q)
    U <- U - mu*(FR%*%E+2*lambda1*U+lambda2*(Lw+t(Lw))%*%U)
    E <- E - mu*(t(FR)%*%U + 2*lambda1*E+2*lambda3*H%*%t(H)%*%E)
    Qhat<-U%*%t(E)
    curObsMAE <- omegaMAE(omega,Q,Qhat)
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
  res<-(list(observedMAE=omegaMAE(accMask,Q,Qhat),
             unobservedMAE=omegaMAE(unobsaccMask,Q,Qhat),
             observedMAPE=omegaMAPE(accMask,Q,Qhat),
             unobservedMAPE=omegaMAPE(unobsaccMask,Q,Qhat),
             converged=converged))
  if(returnQhat) res[['Qhat']]=U%*%t(E)
  return(res)
}

omegaMAE<-function(omega,Q,Qhat){
  return(sum(hadamard.prod(omega,abs(Q-Qhat)))/sum(omega>0))
}
omegaMAPE<-function(omega,Q,Qhat){
  return(sum(hadamard.prod(omega,abs(Q-Qhat)/(Q+0.0001)))/sum(omega>0))
}

enhanceOmega<-function(omega, observable){
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

randomOmega<-function(N,T, sparsity, Q){
  obs<-matrix(0,N,T)
  rownames(obs)<-rownames(Q)
  for (t in 1:T){
    for (n in 1:N){
      s<-rbinom(1, size = round(Q[n,t]), prob = sparsity)
      obs[n,t]<-(s>0)
    }
  }
  return(obs)
}