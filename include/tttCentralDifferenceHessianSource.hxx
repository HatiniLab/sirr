/*
 * tttCentralDifferenceHessianSource.hxx
 *
 *  Created on: Nov 14, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_HXX_
#define INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_HXX_

#include "tttCentralDifferenceHessianSource.h"
#include <itkNthElementImageAdaptor.h>
#include <itkImageAlgorithm.h>
#include <itkConstantPadImageFilter.h>
#include <itkCyclicShiftImageFilter.h>
#include <itkComposeImageFilter.h>
#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#if 0
template<class TReal ,int dim >
typename itk::DataObject::Pointer ttt::CentralDifferenceHessianSource<TReal,dim>::MakeOutput(unsigned int idx) {
	typename itk::DataObject::Pointer output;

	switch (idx) {
	case 0:
		output = (HessianFilterImageType::New()).GetPointer();
		break;
	case 1:
		output = (EnergyImageType::New()).GetPointer();
		break;
	default:
		std::cerr << "No output " << idx << std::endl;
		output = NULL;
		break;
	}
	return output.GetPointer();
}
#endif

template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::HessianKernelToHessianOperator(const typename KernelImageType::Pointer & kernel, typename FilterImageType::Pointer & op){

	//typename TInputImage::ConstPointer input = this->GetInput();

	typename KernelImageType::SizeType kernelSize = kernel->GetLargestPossibleRegion().GetSize();

	typename KernelImageType::SizeType padSize =	 this->m_PadSize;

	typename KernelImageType::SizeType kernelUpperBound;
	typename KernelImageType::SizeType kernelLowerBound=this->m_LowerPad;

	for (unsigned int i = 0; i < dim; ++i) {
		kernelUpperBound[i] = (padSize[i] - kernelSize[i] -this->m_LowerPad[i]);
	}

	// Pad the kernel image with zeros.
	typedef itk::ConstantPadImageFilter<KernelImageType, KernelImageType> KernelPadType;
	typedef typename KernelPadType::Pointer KernelPadPointer;
	KernelPadPointer kernelPadder = KernelPadType::New();
	kernelPadder->SetConstant(itk::NumericTraits<double>::ZeroValue());
	kernelPadder->SetPadUpperBound(kernelUpperBound);
	kernelPadder->SetPadLowerBound(kernelLowerBound);
	kernelPadder->SetNumberOfThreads(this->GetNumberOfThreads());
	kernelPadder->SetInput(kernel);
	//kernelPadder->ReleaseDataFlagOn();


	//progress->RegisterInternalFilter(kernelPadder, 0);
	//paddingWeight * progressWeight );
	kernelPadder->Update();


	//typedef itk::ComplexToModulusImageAdaptor<TFilterImage,double> ModulusType;

	//typename ModulusType::Pointer modulus = ModulusType::New();
#if 0
	typedef itk::ImageFileWriter<KernelImageType> ImageWriter;
	typename ImageWriter::Pointer filterWriter=ImageWriter::New();

	filterWriter->SetInput(k);
	filterWriter->SetFileName("hessianxx.mha");
	filterWriter->Update();
#endif

	typedef itk::CyclicShiftImageFilter<KernelImageType, KernelImageType> KernelShiftFilterType;
	typename KernelShiftFilterType::Pointer kernelShifter =
			KernelShiftFilterType::New();
	typename KernelShiftFilterType::OffsetType kernelShift;
	for (unsigned int i = 0; i < dim; ++i) {
		kernelShift[i] = -(kernelSize[i] / 2)-this->m_LowerPad[i];
	}
	kernelShifter->SetShift(kernelShift);
	kernelShifter->SetNumberOfThreads(this->GetNumberOfThreads());
	kernelShifter->SetInput(kernelPadder->GetOutput());
	kernelShifter->ReleaseDataFlagOn();

	kernelShifter->Update();
	typename KernelImageType::Pointer shifted = kernelShifter->GetOutput();
#if 0

	//paddedKernelImage->DisconnectPipeline();
	{
			std::cout << "Kernel" << std::endl;
			std::cout << shifted << std::endl;
			itk::Index<3> index;

			int R=  shifted->GetBufferedRegion().GetIndex(0);
			int Rmax= shifted->GetBufferedRegion().GetSize(0);
			int C=  shifted->GetBufferedRegion().GetIndex(1);
			int Cmax=  shifted->GetBufferedRegion().GetSize(1);
			int H=  shifted->GetBufferedRegion().GetIndex(2);
			int Hmax=  shifted->GetBufferedRegion().GetSize(2);
			for(int h=H;h<H+Hmax;h++){
				index[2]=h;
				for(int r=R;r<R+Rmax;r++){
					index[0]=r;
					for(int c=C;c<C+Cmax;c++){
						index[1]=c;
						std::cout << shifted->GetPixel(index) << " ";
					}
					std::cout << std::endl;
				}
				std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
			}
			}

#endif
	typedef itk::RealToHalfHermitianForwardFFTImageFilter< KernelImageType,FilterImageType > FFTFilterType;

	typename FFTFilterType::Pointer kernelFFTFilter = FFTFilterType::New();
	kernelFFTFilter->SetNumberOfThreads(this->GetNumberOfThreads());
	kernelFFTFilter->SetInput(kernelShifter->GetOutput());
	//progress->RegisterInternalFilter(kernelFFTFilter, 0.699f * progressWeight);
	kernelFFTFilter->Update();


	op = kernelFFTFilter->GetOutput();
	op->DisconnectPipeline();


}


template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianXX(){


	typedef typename KernelImageType::RegionType KernelRegionType;
	typedef typename KernelImageType::SizeType KernelSizeType;
	typedef typename KernelImageType::IndexType KernelIndexType;
	typename KernelImageType::Pointer kernel = KernelImageType::New();

	KernelRegionType kernelRegion = kernel->GetLargestPossibleRegion();
	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);
	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);
	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);

#if 0
	KernelIndexType index;
	index[0]=0;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,1.0/(m_Spacing[0]*m_Spacing[0]));

	index[0]=1;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,-2.0/(m_Spacing[0]*m_Spacing[0]));

	index[0]=2;
	index[1]=2;
	index[2]=2;

	kernel->SetPixel(index,1.0/(m_Spacing[0]*m_Spacing[0]));
#endif

	KernelIndexType index;
	index[0]=1;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,1.0/(m_Spacing[0]*m_Spacing[0]));

	index[0]=2;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,-2.0/(m_Spacing[0]*m_Spacing[0]));

	index[0]=3;
	index[1]=2;
	index[2]=2;

	kernel->SetPixel(index,1.0/(m_Spacing[0]*m_Spacing[0]));

	this->HessianKernelToHessianOperator(kernel,m_Filters[0]);

	//m_ComplexWeights[0]=1.0;

}

template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianXY(){


	typedef typename KernelImageType::RegionType KernelRegionType;
	typedef typename KernelImageType::SizeType KernelSizeType;
	typedef typename KernelImageType::IndexType KernelIndexType;
	typename KernelImageType::Pointer kernel = KernelImageType::New();

	KernelRegionType kernelRegion = kernel->GetLargestPossibleRegion();

	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);
	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);
	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);
#if 0
	KernelIndexType index;
	index[0] = 1;
	index[1] = 1;
	index[2] = 2;
	kernel->SetPixel(index, 1.0/(m_Spacing[0]*m_Spacing[1]));

	index[0] = 1;
	index[1] = 2;
	index[2] = 2;
	kernel->SetPixel(index, -1.0/(m_Spacing[0]*m_Spacing[1]));

	index[0] = 2;
	index[1] = 1;
	index[2] = 2;
	kernel->SetPixel(index, -1.0/(m_Spacing[0]*m_Spacing[1]));

	index[0] = 2;
	index[1] = 2;
	index[2] = 2;
	kernel->SetPixel(index, 1.0/(m_Spacing[0]*m_Spacing[1]));

#endif

	KernelIndexType index;
	index[0]=1;
	index[1]=1;
	index[2]=2;
	kernel->SetPixel(index,1.0/(4*m_Spacing[0]*m_Spacing[1]));

	index[0]=1;
	index[1]=3;
	index[2]=2;
	kernel->SetPixel(index,-1.0/(4*m_Spacing[0]*m_Spacing[1]));

	index[0]=3;
	index[1]=1;
	index[2]=2;

	kernel->SetPixel(index,-1.0/(4*m_Spacing[0]*m_Spacing[1]));

	index[0]=3;
	index[1]=3;
	index[2]=2;

	kernel->SetPixel(index,1.0/(4*m_Spacing[0]*m_Spacing[1]));


	this->HessianKernelToHessianOperator(kernel,this->m_Filters[1]);
	//m_ComplexWeights[1]=2.0;

}
template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::AllocateOutputs(){

	typename HessianFilterImageType::Pointer hessian = this->GetOutput();
	//typename EnergyImageType::Pointer energy = this->GetOutput2();
	typename HessianFilterImageType::RegionType region;

	for(int i=0;i<dim;i++){
		region.SetIndex(i,-m_LowerPad[i]);
		region.SetSize(i,m_PadSize[i]);
	}
	hessian->SetRegions(region);
	hessian->Allocate();
#if 0
	energy->SetRegions(region);
	energy->Allocate();
#endif

}
template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianXZ(){


	typedef typename KernelImageType::RegionType KernelRegionType;
	typedef typename KernelImageType::SizeType KernelSizeType;
	typedef typename KernelImageType::IndexType KernelIndexType;
	typename KernelImageType::Pointer kernel = KernelImageType::New();

	KernelRegionType kernelRegion = kernel->GetLargestPossibleRegion();

	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);
	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);
	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);
#if 0
	KernelIndexType index;
	index[0] = 1;
	index[1] = 2;
	index[2] = 1;
	kernel->SetPixel(index, 1.0/(m_Spacing[0]*m_Spacing[2]));

	index[0] = 1;
	index[1] = 2;
	index[2] = 2;
	kernel->SetPixel(index, -1.0/(m_Spacing[0]*m_Spacing[2]));

	index[0] = 2;
	index[1] = 2;
	index[2] = 1;
	kernel->SetPixel(index, -1.0/(m_Spacing[0]*m_Spacing[2]));

	index[0] = 2;
	index[1] = 2;
	index[2] = 2;
	kernel->SetPixel(index, 1.0/(m_Spacing[0]*m_Spacing[2]));
#endif


	KernelIndexType index;
	index[0]=1;
	index[1]=2;
	index[2]=1;
	kernel->SetPixel(index,1.0/(4*m_Spacing[0]*m_Spacing[2]));

	index[0]=1;
	index[1]=2;
	index[2]=3;
	kernel->SetPixel(index,-1.0/(4*m_Spacing[0]*m_Spacing[2]));

	index[0]=3;
	index[1]=2;
	index[2]=1;

	kernel->SetPixel(index,-1.0/(4*m_Spacing[0]*m_Spacing[2]));

	index[0]=3;
	index[1]=2;
	index[2]=3;

	kernel->SetPixel(index,1.0/(4*m_Spacing[0]*m_Spacing[2]));

	this->HessianKernelToHessianOperator(kernel,this->m_Filters[2]);
	//m_ComplexWeights[2]=2.0;

}


template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianYY(){


	typename KernelImageType::Pointer kernel = KernelImageType::New();
	typename KernelImageType::RegionType kernelRegion = kernel->GetLargestPossibleRegion();
	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);
	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);

	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);
#if 0

	typename KernelImageType::IndexType index;
	index[0]=2;
	index[1]=0;
	index[2]=2;
	kernel->SetPixel(index,1.0/(m_Spacing[1]*m_Spacing[1]));

	index[0]=2;
	index[1]=1;
	index[2]=2;
	kernel->SetPixel(index,-2.0/(m_Spacing[1]*m_Spacing[1]));

	index[0]=2;
	index[1]=2;
	index[2]=2;

	kernel->SetPixel(index,1.0/(m_Spacing[1]*m_Spacing[1]));
#endif


	typename KernelImageType::IndexType index;
	index[0]=2;
	index[1]=1;
	index[2]=2;
	kernel->SetPixel(index,1.0/(m_Spacing[1]*m_Spacing[1]));

	index[0]=2;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,-2.0/(m_Spacing[1]*m_Spacing[1]));

	index[0]=2;
	index[1]=3;
	index[2]=2;

	kernel->SetPixel(index,1.0/(m_Spacing[1]*m_Spacing[1]));

	this->HessianKernelToHessianOperator(kernel,this->m_Filters[3]);
	//m_ComplexWeights[3]=1.0;

}

template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianYZ(){


	typedef typename KernelImageType::RegionType KernelRegionType;
	typedef typename KernelImageType::SizeType KernelSizeType;
	typedef typename KernelImageType::IndexType KernelIndexType;
	typename KernelImageType::Pointer kernel = KernelImageType::New();

	KernelRegionType kernelRegion = kernel->GetLargestPossibleRegion();

	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);

	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);

	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);

#if 0
	KernelIndexType index;
	index[0] = 2;
	index[1] = 1;
	index[2] = 1;
	kernel->SetPixel(index, 1.0/(m_Spacing[1]*m_Spacing[2]));

	index[0] = 2;
	index[1] = 1;
	index[2] = 2;
	kernel->SetPixel(index, -1.0/(m_Spacing[1]*m_Spacing[2]));

	index[0] = 2;
	index[1] = 2;
	index[2] = 1;
	kernel->SetPixel(index, -1.0/(m_Spacing[1]*m_Spacing[2]));

	index[0] = 2;
	index[1] = 2;
	index[2] = 2;
	kernel->SetPixel(index, 1.0/(m_Spacing[1]*m_Spacing[2]));
#endif


	KernelIndexType index;
	index[0]=1;
	index[1]=1;
	index[2]=2;
	kernel->SetPixel(index,1.0/(4*m_Spacing[1]*m_Spacing[2]));

	index[0]=1;
	index[1]=3;
	index[2]=2;
	kernel->SetPixel(index,-1.0/(4*m_Spacing[1]*m_Spacing[2]));

	index[0]=3;
	index[1]=1;
	index[2]=2;

	kernel->SetPixel(index,-1.0/(4*m_Spacing[1]*m_Spacing[2]));

	index[0]=3;
	index[1]=3;
	index[2]=2;

	kernel->SetPixel(index,1.0/(4*m_Spacing[1]*m_Spacing[2]));
	this->HessianKernelToHessianOperator(kernel,this->m_Filters[4]);
	//m_ComplexWeights[4]=2.0;

}

template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::InitHessianZZ(){


	typename KernelImageType::Pointer kernel = KernelImageType::New();
	typename KernelImageType::RegionType kernelRegion = kernel->GetLargestPossibleRegion();
	kernelRegion.SetIndex(0, 0);
	kernelRegion.SetIndex(1, 0);
	kernelRegion.SetIndex(2, 0);
	kernelRegion.SetSize(0, 5);
	kernelRegion.SetSize(1, 5);
	kernelRegion.SetSize(2, 5);

	kernel->SetRegions(kernelRegion);

	kernel->Allocate();
	kernel->FillBuffer(0);
#if 0
	typename KernelImageType::IndexType index;
	index[0]=2;
	index[1]=2;
	index[2]=0;
	kernel->SetPixel(index,1.0/(m_Spacing[2]*m_Spacing[2]));

	index[0]=2;
	index[1]=2;
	index[2]=1;
	kernel->SetPixel(index,-2.0/(m_Spacing[2]*m_Spacing[2]));

	index[0]=2;
	index[1]=2;
	index[2]=2;

	kernel->SetPixel(index,1.0/(m_Spacing[2]*m_Spacing[2]));
#endif

	typename KernelImageType::IndexType index;
	index[0]=2;
	index[1]=2;
	index[2]=1;
	kernel->SetPixel(index,1.0/(m_Spacing[2]*m_Spacing[2]));

	index[0]=2;
	index[1]=2;
	index[2]=2;
	kernel->SetPixel(index,-2.0/(m_Spacing[2]*m_Spacing[2]));

	index[0]=2;
	index[1]=2;
	index[2]=3;

	kernel->SetPixel(index,1.0/(m_Spacing[2]*m_Spacing[2]));
	this->HessianKernelToHessianOperator(kernel,this->m_Filters[5]);
	//m_ComplexWeights[5]=1.0;
}
#if 0
template<class TReal, int dim> void ttt::CentralDifferenceHessianSource<TReadl,dim>::CreateGaussianDerivative(itk::Vector<unsigned,dim> & order){

}
#endif

template<class TReal ,int dim > void ttt::CentralDifferenceHessianSource<TReal,dim>::GenerateData(){
	if(dim==3){
		this->AllocateOutputs();
		m_Filters.resize(6);
		this->InitHessianXX();
		this->InitHessianXY();
		this->InitHessianXZ();
		this->InitHessianYY();
		this->InitHessianYZ();
		this->InitHessianZZ();


		typedef itk::ComposeImageFilter<FilterImageType, HessianFilterImageType> ComposeCovariantVectorImageFilterType;



		typename ComposeCovariantVectorImageFilterType::Pointer composeResultFilter = ComposeCovariantVectorImageFilterType::New();
		composeResultFilter->SetNumberOfThreads(this->GetNumberOfThreads());

		for(unsigned int i=0;i<6;i++){
			composeResultFilter->SetInput(i,this->m_Filters[i]);
		}
		composeResultFilter->Update();
		this->SetPrimaryOutput(composeResultFilter->GetOutput());

#if 0
		typedef itk::NthElementImageAdaptor<HessianFilterImageType,std::complex<double> > NthElementImageAdaptorType;
		typename NthElementImageAdaptorType::Pointer nthadaptor=NthElementImageAdaptorType::New();
		nthadaptor->SetImage(this->GetOutput());
		for(int i=0;i<m_Filters.size();i++){
			nthadaptor->SelectNthElement(i);
			itk::ImageRegionConstIterator<FilterImageType> it(this->m_Filters[i],this->m_Filters[i]->GetRequestedRegion());
			itk::ImageRegionIterator<NthElementImageAdaptorType> ot(nthadaptor,nthadaptor->GetRequestedRegion());
			while(!it.IsAtEnd()){
				ot.Set(it.Get());
				++ot;
				++it;
			}
		}
#endif
		m_Filters.clear();
	}
	std::cout << "Hessian Initialized" << std::endl;

}


#endif /* INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_HXX_ */
