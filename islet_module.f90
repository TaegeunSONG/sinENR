
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


! =====================================================================
!  Pseudo-random number utilities.
!
!  These replace the Numerical Recipes ran1 / gasdev / ran_seed of the
!  original source, which are copyrighted and may not be redistributed.
!  The external interface is unchanged -- ran1(idum) returns a uniform
!  deviate on [0,1) and advances the state held in idum, gasdev(idum)
!  returns a standard normal deviate, and ran_seed(idum) sets idum from
!  the wall clock -- so no call site needed editing.
!
!  Uniform deviates come from Marsaglia's xorshift64 (J. Stat. Softw. 8,
!  14 (2003)): shifts and exclusive-ors only, so nothing can overflow.
!  The top 53 bits of the state are scaled into [0,1).  Normal deviates
!  use the polar method of Marsaglia and Bray (SIAM Rev. 6, 260 (1964)),
!  here in double precision, where the original used single-precision
!  temporaries.
!
!  This is not the generator the original Fortran used, so a trajectory
!  from this file is statistically equivalent to, but not bit-identical
!  with, one from the original -- the same caveat the manuscript records
!  for the Python port in islet_model.py.
! =====================================================================

FUNCTION ran1(idum)
implicit none
integer*8 :: idum
real*8 :: ran1
integer*8 :: x
integer*8, parameter :: fallback = 4101842887655102017_8
real*8, parameter :: two53 = 9007199254740992.0d0

x = idum
if (x == 0_8) x = fallback            ! xorshift cannot escape a zero state

x = ieor(x, ishft(x,  13))
x = ieor(x, ishft(x,  -7))
x = ieor(x, ishft(x,  17))

idum = x
ran1 = real(ishft(x, -11), 8) / two53
END FUNCTION ran1


FUNCTION gasdev(idum)
implicit none
integer*8 :: idum
real*8 :: gasdev
real*8, external :: ran1
real*8 :: u1, u2, s, f
real*8, save :: spare = 0.0d0
logical, save :: have_spare = .false.

if (have_spare) then
  have_spare = .false.
  gasdev = spare
  return
end if

s = 0.0d0
do while (s >= 1.0d0 .or. s <= 0.0d0)
  u1 = 2.0d0*ran1(idum) - 1.0d0
  u2 = 2.0d0*ran1(idum) - 1.0d0
  s  = u1*u1 + u2*u2
end do

f = sqrt(-2.0d0*log(s)/s)
spare = u1*f
have_spare = .true.
gasdev = u2*f
END FUNCTION gasdev


subroutine ran_seed(idum)
implicit none
integer*8, intent(out) :: idum
integer, dimension(8) :: t
call date_and_time(values=t)
idum = int(t(5),8)*3600000_8 + int(t(6),8)*60000_8 &
     + int(t(7),8)*1000_8    + int(t(8),8) + 1_8
end subroutine ran_seed

      
