!Fortran subroutine for the calculation of the 4th order shapiro filter
!    Copyright (C) 2018  Hans Brenna
!    This program is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!    This program is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!    You should have received a copy of the GNU General Public License
!    along with this program. If not, see <http://www.gnu.org/licenses/>.

subroutine fourth_order_shapiro_filter(psi,vector,tau,psi_filtered,nlambda,nphi,nP)
	!===============================================
	!apply the 4th order shapiro filter to data. vector
	! specifies whether values from across the pole
	!shall have a sign reversal
	!===============================================
	implicit none
	integer, intent(in)    :: nlambda,nphi,nP
	logical, intent(in)    :: vector
	real(8), intent(in)    :: tau
	real(8), intent(in), dimension(0:nlambda-1,0:nphi-1,0:nP-1) :: psi
	real(8), intent(out), dimension(0:nlambda-1,0:nphi-1,0:nP-1) :: psi_filtered
	real(8), dimension(0:nlambda-1,0:nphi-1,0:nP-1) :: psi_filtered_lambda
	real(8), dimension(-4:nlambda+3,0:nphi-1,0:nP-1)    :: psi_d
	real(8), dimension(0:nlambda-1,-4:nphi+3,0:nP-1)    :: psi_filtered_lambda_d
	integer :: i,j,k
	real(8) :: vect

	!configure ghost cells
	psi_d = 0.0
	psi_filtered_lambda_d = 0.0
	!periodic boundary in lambda-direction
	psi_d(0:nlambda-1,:,:) = psi
	psi_d(-1,:,:) = psi(nlambda-1,:,:)
	psi_d(-2,:,:) = psi(nlambda-2,:,:)
	psi_d(-3,:,:) = psi(nlambda-3,:,:)
	psi_d(-4,:,:) = psi(nlambda-4,:,:)
	psi_d(nlambda,:,:) = psi(0,:,:)
	psi_d(nlambda+1,:,:) = psi(1,:,:)
	psi_d(nlambda+2,:,:) = psi(2,:,:)
	psi_d(nlambda+3,:,:) = psi(3,:,:)

	!Filter in lambda-direction:
	do k = 0,nP-1
		do j = 0,nphi-1
			do i = 0,nlambda-1
				psi_filtered_lambda(i,j,k) = ((1.0/256.0)*(186.0*psi_d(i,j,k)+56.0*(psi_d(i-1,j,k)&
                           +psi_d(i+1,j,k))-28.0*(psi_d(i-2,j,k)+psi_d(i+2,j,k))&
                           +8.0*(psi_d(i-3,j,k)+psi_d(i+3,j,k))-(psi_d(i-4,j,k)+psi_d(i+4,j,k))))
			enddo
		enddo
	enddo
	!polar condition
	if (vector) then
		vect = -1.0
	else
		vect = 1.0
	end if
	psi_filtered_lambda_d(:,0:nphi-1,:) = psi_filtered_lambda
	psi_filtered_lambda_d(:,-1,:) =  vect*CSHIFT(psi_filtered_lambda(:,0,:),nlambda/2,1)
	psi_filtered_lambda_d(:,-2,:) =  vect*CSHIFT(psi_filtered_lambda(:,1,:),nlambda/2,1)
	psi_filtered_lambda_d(:,-3,:) =  vect*CSHIFT(psi_filtered_lambda(:,2,:),nlambda/2,1)
	psi_filtered_lambda_d(:,-4,:) =  vect*CSHIFT(psi_filtered_lambda(:,3,:),nlambda/2,1)
	psi_filtered_lambda_d(:,nphi,:) = vect*CSHIFT(psi_filtered_lambda(:,nphi-1,:),nlambda/2,1)
	psi_filtered_lambda_d(:,nphi+1,:) = vect*CSHIFT(psi_filtered_lambda(:,nphi-2,:),nlambda/2,1)
	psi_filtered_lambda_d(:,nphi+2,:) = vect*CSHIFT(psi_filtered_lambda(:,nphi-3,:),nlambda/2,1)
	psi_filtered_lambda_d(:,nphi+3,:) = vect*CSHIFT(psi_filtered_lambda(:,nphi-4,:),nlambda/2,1)

!filter in $\phi$-direction
	do k = 0,nP-1
		do j = 0,nphi-1
			do i = 0,nlambda-1
				psi_filtered(i,j,k) = ((1.0/256.0)*(186.0*psi_filtered_lambda_d(i,j,k)+56.0*(psi_filtered_lambda_d(i,j-1,k)&
                            +psi_filtered_lambda_d(i,j+1,k))-28.0*(psi_filtered_lambda_d(i,j-2,k)+psi_filtered_lambda_d(i,j+2,k))&
                            +8.0*(psi_filtered_lambda_d(i,j-3,k)+psi_filtered_lambda_d(i,j+3,k))-(psi_filtered_lambda_d(i,j-4,k)&
                            +psi_filtered_lambda_d(i,j+4,k))))
			enddo
		enddo
	enddo

	psi_filtered = psi-1/tau*(psi-psi_filtered)
end subroutine
