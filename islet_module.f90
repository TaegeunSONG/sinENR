
subroutine trace(g,amp,theta,pam,nmax,dmax,itmax,isample,kmat,om,gdata,hdata)
Implicit none
integer*8, intent(in) :: nmax, dmax, itmax, isample
real*8, intent(in) :: pam(9), kmat(3,3), om(3,nmax)
real*8, intent(inout) :: g, amp(3,nmax), theta(3,nmax) 
real*8, intent(out) :: gdata(dmax), hdata(3,nmax)
integer*8 n, i, j, it, icount
real*8 g_new, pi
real*8 amp_new(3,nmax), theta_new(3,nmax)

pi=4.*atan(1.) 

icount=0 ; gdata=0. ; hdata=0.
do it=1, itmax
 do i=1, 3
  do n=1, nmax
   if(amp(i,n).le.0.)then
    amp(i,n)=-amp(i,n) ; theta(i,n)=theta(i,n)+pi
   endif
  enddo
 enddo
   
 call rk4(pam,kmat,om,nmax,amp,theta,g,amp_new,theta_new,g_new)
  
 if(it/isample*isample.eq.it)then
 icount=icount+1
 gdata(icount)=g
  do n=1, nmax
  hdata(1,icount)=hdata(1,icount)+amp(1,n)*(1.+cos(theta(1,n)))
  hdata(2,icount)=hdata(2,icount)+amp(2,n)*(1.+cos(theta(2,n)))
  hdata(3,icount)=hdata(3,icount)+amp(3,n)*(1.+cos(theta(3,n)))
  enddo
 endif

 g=g_new
 do i=1, 3
  do n=1, nmax
  amp(i,n)=amp_new(i,n)
  theta(i,n)=theta_new(i,n)
  enddo
 enddo

enddo !it

return
end

subroutine rk4(pam,kmat,om,nmax,amp,theta,g,amp_new,theta_new,g_new)
implicit none
integer*8, intent(in) :: nmax
real*8, intent(in) :: pam(9), kmat(3,3), om(3,nmax), amp(3,nmax), theta(3,nmax), g
real*8, intent(out) :: amp_new(3,nmax), theta_new(3,nmax), g_new
integer*8 i, j, n
real*8 damp(3,nmax), dtheta(3,nmax), dg
real*8 k1g,k2g,k3g,k4g
real*8 k1a(3,nmax), k2a(3,nmax), k3a(3,nmax), k4a(3,nmax)
real*8 k1p(3,nmax), k2p(3,nmax), k3p(3,nmax), k4p(3,nmax)

k1g=0. ; k2g=0. ; k3g=0. ; k4g=0.
k1a=0. ; k2a=0. ; k3a=0. ; k4a=0.
k1p=0. ; k2p=0. ; k3p=0. ; k4p=0.

call derivs(pam,kmat,om,nmax,g,amp,theta,damp,dtheta,dg)
k1g=pam(1)*dg
do i=1, 3
 do n=1, nmax
 k1a(i,n)=pam(1)*damp(i,n)
 k1p(i,n)=pam(1)*dtheta(i,n)
 enddo
enddo

g_new=g+0.5*k1g
do i=1, 3
 do n=1, nmax
 amp_new(i,n)=amp(i,n)+0.5*k1a(i,n)
 theta_new(i,n)=theta(i,n)+0.5*k1p(i,n)
 enddo
enddo
call derivs(pam,kmat,om,nmax,g_new,amp_new,theta_new,damp,dtheta,dg)
k2g=pam(1)*dg
do i=1, 3
 do n=1, nmax
 k2a(i,n)=pam(1)*damp(i,n)
 k2p(i,n)=pam(1)*dtheta(i,n)
 enddo
enddo

g_new=g+0.5*k2g
do i=1, 3
 do n=1, nmax
 amp_new(i,n)=amp(i,n)+0.5*k2a(i,n)
 theta_new(i,n)=theta(i,n)+0.5*k2p(i,n)
 enddo
enddo
call derivs(pam,kmat,om,nmax,g_new,amp_new,theta_new,damp,dtheta,dg)
k3g=pam(1)*dg
do i=1, 3
 do n=1, nmax
 k3a(i,n)=pam(1)*damp(i,n)
 k3p(i,n)=pam(1)*dtheta(i,n)
 enddo
enddo

g_new=g+k3g
do i=1, 3
 do n=1, nmax
 amp_new(i,n)=amp(i,n)+k3a(i,n)
 theta_new(i,n)=theta(i,n)+k3p(i,n)
 enddo
enddo
call derivs(pam,kmat,om,nmax,g_new,amp_new,theta_new,damp,dtheta,dg)
k4g=pam(1)*dg
do i=1, 3
 do n=1, nmax
 k4a(i,n)=pam(1)*damp(i,n)
 k4p(i,n)=pam(1)*dtheta(i,n)
 enddo
enddo

g_new=g+(k1g+2.*(k2g+k3g)+k4g)/6.
do i=1, 3
 do n=1, nmax
 amp_new(i,n)=amp(i,n)+(k1a(i,n)+2.*(k2a(i,n)+k3a(i,n))+k4a(i,n))/6.
 theta_new(i,n)=theta(i,n)+(k1p(i,n)+2.*(k2p(i,n)+k3p(i,n))+k4p(i,n))/6.
 enddo
enddo


return
end

subroutine derivs(pam,kmat,om,nmax,g,amp,theta,damp,dtheta,dg)
implicit none
integer*8, intent(in) :: nmax
real*8, intent(in) :: pam(9), kmat(3,3), om(3,nmax), g, amp(3,nmax), theta(3,nmax)
real*8, intent(out) :: dg, dtheta(3,nmax), damp(3,nmax)
integer*8 i, j, n
real*8 g0, ha, hb, rtmp1, rtmp2, glures, inres, lamda
real*8 Gam, lamdmax, fslop, delg
real*8 mu(3), lam(3)

Gam=pam(2) ; lamda=pam(3) ; lamdmax=pam(4) ; g0=pam(5) ; fslop=pam(6) 
delg=pam(7) ; glures=pam(8) ; inres=pam(9)

lam=0.
lam(1)= 0.5*(1.+tanh((-g+g0)/fslop))   ! lamda alpha
lam(2)= 0.5*(1.+tanh((g-g0)/fslop))    ! lamda beta
lam(3)=lamdmax*0.5*(1.+tanh((g-g0+delg)/fslop))    ! lamda delta

mu=0.
mu(1)=Gam*(g-g0)
mu(2)=Gam*(g0-g)
mu(3)=Gam*(g0-g)

 do n=1, nmax
 do i=1, 3
   rtmp1=0. ; rtmp2=0.
   do j=1, 3
   rtmp1=rtmp1+kmat(j,i)*amp(j,n)*cos(theta(j,n)-theta(i,n))
   if(amp(i,n).ne.0.) rtmp2=rtmp2+kmat(j,i)*amp(j,n)*sin(theta(j,n)-theta(i,n))/amp(i,n)
   enddo
  damp(i,n)=(lam(i)-abs(amp(i,n))**2)*amp(i,n)+rtmp1
  dtheta(i,n)=om(i,n)+mu(i)*cos(theta(i,n))+rtmp2
 enddo
 enddo

 ha=0.;hb=0.
 do n=1, nmax
 ha=ha+amp(1,n)*(cos(theta(1,n))+1.)
 hb=hb+amp(2,n)*(cos(theta(2,n))+1.)
 enddo

 dg=lamda*((1.-glures)*ha*g0-(1.-inres)*hb*g)

return
end


FUNCTION gasdev(idum)
INTEGER*8 idum
REAL*8 gasdev, ran1
INTEGER iset
REAL fac,gset,rsq,v1,v2
SAVE iset,gset
DATA iset/0/
if (iset==0) then
  do
    v1=2.*ran1(idum)-1.
    v2=2.*ran1(idum)-1.
    rsq=v1**2+v2**2
    if(.NOT.(rsq>=1..or.rsq==0.)) EXIT
  end do
  fac=sqrt(-2.*log(rsq)/rsq)
  gset=v1*fac
  gasdev=v2*fac
  iset=1
else
  gasdev=gset
  iset=0
endif
END FUNCTION gasdev


FUNCTION ran1(idum)
INTEGER*8 idum
INTEGER IA,IM,IQ,IR,NTAB,NDIV
REAL*8 ran1
REAL AM,EPS,RNMX
PARAMETER (IA=16807,IM=2147483647,AM=1./IM,IQ=127773,IR=2836)
PARAMETER (NTAB=32,NDIV=1+(IM-1)/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
INTEGER j,k,iv(NTAB),iy
SAVE iv,iy
DATA iv /NTAB*0/, iy /0/
if (idum.le.0.or.iy.eq.0) then
!  idum=max(-idum,1)
  idum=max(idum,1)
  do 11 j=NTAB+8,1,-1
   k=idum/IQ
   idum=IA*(idum-k*IQ)-IR*k
    if (idum.lt.0) idum=idum+IM
    if (j.le.NTAB) iv(j)=idum
11   continue
  iy=iv(1)
endif
k=idum/IQ
idum=IA*(idum-k*IQ)-IR*k
if (idum.lt.0) idum=idum+IM
j=1+iy/NDIV
iy=iv(j)
iv(j)=idum
ran1=min(AM*iy,RNMX)
return
END

subroutine ran_seed(idum)
implicit none
integer*8 idum
integer,DIMENSION(8) :: time
integer,DIMENSION(50) :: seed
CALL DATE_AND_TIME(values=time)     ! Get the current time
seed(1) = time(4)*(360000*time(5) + 6000*time(6) + 100*time(7) + time(8))
CALL RANDOM_SEED(PUT=seed)
idum=seed(1)
end
      
