subroutine prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,omegas_n,Fl,fcor,g,a,cosphi,tanphi,phi,dlambda,dphi,dP,Pf,&
	dt,nlambda,nphi,nP)
!======================================================================
!Update prognostic velocity u in zonal direction
!======================================================================
	implicit none
	integer, intent(in)     :: nlambda,nphi,nP
	real(8), intent(in)     :: dlambda,dphi,dP,dt,g,a
	real(8), intent(in), dimension(0:nP-1)     :: Pf
	real(8), intent(in), dimension(0:nphi-1)     :: fcor,cosphi,tanphi,phi
	real(8), intent(in), dimension(0:nlambda-1, 0:nphi-1, 0:nP-1)      :: u_n,u_p,v_n,omega_n,z_n,Fl
	real(8), intent(in), dimension(0:nlambda-1, 0:nphi-1)              :: us,Ps_n,omegas_n
	real(8), intent(inout), dimension(0:nlambda-1, 0:nphi-1, 0:nP-1)   :: u_f
!f2py intent(in,out) :: u_f
	!real(8), intent(out), dimension(0:nlambda-1,0:nphi-1,0:nP-1) :: a_adv_uu,a_adv_vu,a_adv_omegau,a_cor_u,a_gdz_u,a_curv_u
	real(8), dimension(-2:nlambda+1, 0:nphi-1, 0:nP-1)     :: u_nd
	real(8), dimension(0:nlambda-1, -2:nphi+1, 0:nP-1)     :: u_np
	real(8), dimension(-2:nlambda+1, 0:nphi-1, 0:nP-1)     :: z_nd
	real(8)                 :: adv_uu,adv_vu,adv_omegau,cor_u,gdz_u,curv_u
	integer  :: i,j,k
	

	! configure ghost cells for boundary conditions on u_n and z_n
	!set core of dummy variables
	u_nd = 0.0
	u_np = 0.0
	z_nd = 0.0
	u_nd(0:nlambda-1,0:nphi-1,:) = u_n
	u_np(0:nlambda-1,0:nphi-1,:) = u_n
	z_nd(0:nlambda-1,:,:) = z_n
	! periodic boundary in $\lambda$-direction
	u_nd(-2,:,:) = u_n(nlambda-2,:,:)
	u_nd(-1,:,:) = u_n(nlambda-1,:,:)
	u_nd(nlambda+1,:,:) = u_n(1,:,:)
	u_nd(nlambda,:,:) = u_n(0,:,:)
	z_nd(-2,:,:) = z_n(nlambda-2,:,:)
	z_nd(-1,:,:) = z_n(nlambda-1,:,:)
	z_nd(nlambda+1,:,:) = z_n(1,:,:)
	z_nd(nlambda,:,:) = z_n(0,:,:)
	!polar boundary condition, see description elsewhere
	u_np(:,-1,:) = -CSHIFT(u_n(:,0,:),nlambda/2,1)
	u_np(:,-2,:) = -CSHIFT(u_n(:,1,:),nlambda/2,1)
	u_np(:,nphi,:) = -CSHIFT(u_n(:,nphi-1,:),nlambda/2,1)
	u_np(:,nphi+1,:) = -CSHIFT(u_n(:,nphi-2,:),nlambda/2,1)

	!calculate tendencies
	do k = 0,nP-1
		do j = 0,nphi-1
			do i = 0,nlambda-1
				adv_uu = (u_nd(i,j,k)/(a*cosphi(j)))*(u_nd(i-2,j,k) &
					+ 8*(-u_nd(i-1,j,k) + u_nd(i+1,j,k)) - u_nd(i+2,j,k))/(12.0*dlambda)
				adv_vu = (v_n(i,j,k)/(a)*((u_np(i,j-2,k) &
					+ 8*(-u_np(i,j-1,k) + u_np(i,j+1,k)) - u_np(i,j+2,k))/(12.0*dphi)))
				if (k == 0) then
					adv_omegau = (((0.5*(u_n(i,j,k+1)+u_n(i,j,k))&
						*omega_n(i,j,k+1)))/(dP) - u_n(i,j,k)*(omega_n(i,j,k+1))/(dP))
				else if (k==nP-1) then
					adv_omegau = (((us(i,j)*omegas_n(i,j)-0.5*(u_n(i,j,k)+u_n(i,j,k-1))&
						*omega_n(i,j,k)))/(Ps_n(i,j)-Pf(nP-1)) - u_n(i,j,k)*(omegas_n(i,j)&
					-omega_n(i,j,k))/(Ps_n(i,j)-Pf(nP-1)))
				else
					 adv_omegau = (((0.5*(u_n(i,j,k+1)+u_n(i,j,k))*omega_n(i,j,k+1)-0.5&
                               *(u_n(i,j,k)+u_n(i,j,k-1))*omega_n(i,j,k)))/(dP)&
                               - u_n(i,j,k)*(omega_n(i,j,k+1)-omega_n(i,j,k))/(dP))
				end if

				cor_u = fcor(j)*v_n(i,j,k)
				gdz_u = (((g)/(a*cosphi(j)))*((z_nd(i-2,j,k)+8*(-z_nd(i-1,j,k)+z_nd(i+1,j,k))&
                      -z_nd(i+2,j,k))/(12.0*dlambda)))
				curv_u = ((u_n(i,j,k)*v_n(i,j,k)*tanphi(j)/a))
				! a_adv_uu(i,j,k) = adv_uu
				! a_adv_vu(i,j,k) = adv_vu
				! a_adv_omegau(i,j,k) = adv_omegau
				! a_cor_u(i,j,k) = cor_u
				! a_gdz_u(i,j,k) = gdz_u
				! a_curv_u(i,j,k) = curv_u
				u_f(i,j,k) = u_p(i,j,k)+2.0*dt*(-(adv_uu+adv_vu+adv_omegau)+cor_u-gdz_u+curv_u-Fl(i,j,k))
			enddo
		enddo
	enddo
end subroutine