SUBROUTINE diag_omega(omega_n,u_n,v_n,cosphi,a,dP,dlambda,dphi,nlambda,nphi,nP)
	!=========================================================================
	!Calculate the diagnostic vertical velocity $\omega$ from the divergence
	!of the horizontal flow field
	!=========================================================================
	implicit none
	real(8), intent(in)     :: a,dP,dlambda,dphi
	integer, intent(in)     :: nlambda,nphi,nP
	real(8), intent(inout), dimension(0:nlambda-1,0:nphi-1,0:nP-1) :: omega_n
!f2py intent(in,out) :: omega_n
	real(8), intent(in), dimension(0:nlambda-1,0:nphi-1,0:nP-1)    :: u_n,v_n
	real(8), intent(in), dimension(0:nphi-1)           :: cosphi
	real(8), dimension(-2:nlambda+1,0:nphi-1,0:nP-1)   :: u_nd
	real(8), dimension(0:nlambda-1,-2:nphi+1,0:nP-1)   :: v_nd
	real(8), dimension(-2:nphi+1)   :: cosphi_d
	real(8)    :: int_div_u,int_div_v !integrands
	integer    :: i,j,k,m !loop variables

	!handle boundary conditions by setting up ghost cells around arrays
	!set core of arrays
	u_nd = 0.0
	v_nd = 0.0
	cosphi_d = 0.0
	u_nd(0:nlambda-1,:,:) = u_n
	v_nd(:,0:nphi-1,:) = v_n
	cosphi_d(0:nphi-1) = cosphi
	!set periodic boundary condition on u_nd
	u_nd(-2,:,:) = u_n(nlambda-2,:,:)
	u_nd(-1,:,:) = u_n(nlambda-1,:,:)
	u_nd(nlambda+1,:,:) = u_n(1,:,:)
	u_nd(nlambda,:,:) = u_n(0,:,:)
	!set polar boundary condition on v_nd and cosphi_d
	v_nd(:,-1,:) = -CSHIFT(v_n(:,0,:),nlambda/2,1)
	v_nd(:,-2,:) = -CSHIFT(v_n(:,1,:),nlambda/2,1)
	v_nd(:,nphi,:) = -CSHIFT(v_n(:,nphi-1,:),nlambda/2,1)
	v_nd(:,nphi+1,:) = -CSHIFT(v_n(:,nphi-2,:),nlambda/2,1)
	cosphi_d(-1) = -cosphi(0)
	cosphi_d(-2) = -cosphi(1)
	cosphi_d(nphi) = -cosphi(nphi-1)
	cosphi_d(nphi+1) = -cosphi(nphi-2)

	omega_n(:,:,0) = 0.0
	!begin loops
	do i = 0,nlambda-1
		do j = 0,nphi-1
			do k = 1,nP-1
				int_div_u = 0.0
				int_div_v = 0.0
				do m = 0,k-1
					int_div_v = int_div_v + (((cosphi_d(j-2)*v_nd(i,j-2,m)+8.0*(-cosphi_d(j-1)&
                               *v_nd(i,j-1,m)+cosphi_d(j+1)*v_nd(i,j+1,m))&
                               -cosphi_d(j+2)*v_nd(i,j+2,m))/(12.0*dphi))*dP)
					int_div_u = int_div_u + (((u_nd(i-2,j,m)+8.0*(-u_nd(i-1,j,m)+u_nd(i+1,j,m))&
                               -u_nd(i+2,j,m))/(12.0*dlambda))*dP)
				enddo
				omega_n(i,j,k) = -(((1.0/(a*cosphi(j)))*int_div_u)+((1.0/(a*cosphi(j)))*int_div_v))
			enddo
		enddo
	enddo
end SUBROUTINE