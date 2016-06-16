/*
 * tttHessianIFFTImageFilter.hxx
 *
 *  Created on: Oct 23, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTHESSIANIFFTIMAGEFILTER_HXX_
#define INCLUDE_TTTHESSIANIFFTIMAGEFILTER_HXX_

#include "tttHessianIFFTImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkComposeImageFilter.h"
#include <itkHalfHermitianToRealInverseFFTImageFilter.h>
template< class TInputImage, class TOutputImage >

void ttt::HessianIFFTImageFilter<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
// call the superclass' implementation of this method
Superclass::GenerateOutputInformation();
// get pointers to the input and output
typename TInputImage::ConstPointer inputPtr = this->GetInput();
typename TOutputImage::Pointer outputPtr = this->GetOutput();
if ( !inputPtr || !outputPtr )
{
return;
}
// This is all based on the same function in itk::ShrinkImageFilter.
// ShrinkImageFilter also modifies the image spacing, but spacing
// has no meaning in the result of an FFT. For an IFFT, since the
// spacing is propagated to the complex result, we can use the spacing
// from the input to propagate back to the output.
const typename TInputImage::SizeType & inputSize =
inputPtr->GetLargestPossibleRegion().GetSize();
const typename TInputImage::IndexType & inputStartIndex =
inputPtr->GetLargestPossibleRegion().GetIndex();
typename TOutputImage::SizeType outputSize;
typename TOutputImage::IndexType outputStartIndex;
// In 4.3.4 of the FFTW documentation, they indicate the size of
// of a real-to-complex FFT is N * N ... + (N /2+1)
// 1 2 d
// complex numbers.
// Going from complex to real, you know the output is at least
// twice the size in the last dimension as the input, but it might
// be 2*size+1. Consequently, you need to check whether the actual
// X dimension is even or odd.

outputSize[0] = ( inputSize[0] - 1 ) * 2;
#if 0
if ( this->GetActualXDimensionIsOdd() )
{
outputSize[0]++;
}
#endif
outputStartIndex[0] = inputStartIndex[0];
for ( unsigned int i = 1; i < TOutputImage::ImageDimension; i++ )
{
outputSize[i] = inputSize[i];
outputStartIndex[i] = inputStartIndex[i];
}
typename TOutputImage::RegionType outputLargestPossibleRegion;
outputLargestPossibleRegion.SetSize( outputSize );
outputLargestPossibleRegion.SetIndex( outputStartIndex );
outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
}
template< class TInputImage, class TOutputImage >
void ttt::HessianIFFTImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
Superclass::GenerateInputRequestedRegion();
// Get pointers to the input and output
typename TInputImage::Pointer inputPtr =
const_cast< TInputImage * >( this->GetInput() );
if ( inputPtr )
{
inputPtr->SetRequestedRegionToLargestPossibleRegion();
}
}
template< class TInputImage, class TOutputImage >
void ttt::HessianIFFTImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *)
{
this->GetOutput()
->SetRequestedRegion( this->GetOutput()->GetLargestPossibleRegion() );
}

template<class TInputImage,class TOutputImage>
void ttt::HessianIFFTImageFilter<TInputImage,TOutputImage>::GenerateData(){

	this->AllocateOutputs();
	typedef itk::Image<double,TInputImage::ImageDimension> InternalType;
	typedef itk::Image<std::complex<double>,TInputImage::ImageDimension> InternalComplexType;

	typedef itk::VectorIndexSelectionCastImageFilter<TInputImage, InternalComplexType> IndexSelectionType;
	typename IndexSelectionType::Pointer indexSelectionFilter = IndexSelectionType::New();

	typedef itk::ComposeImageFilter<InternalType, TOutputImage> ComposeImageFilterType;

	indexSelectionFilter->SetInput(this->GetInput());
	indexSelectionFilter->SetNumberOfThreads(this->GetNumberOfThreads());
	typename ComposeImageFilterType::Pointer composeResultFilter = ComposeImageFilterType::New();
	composeResultFilter->SetNumberOfThreads(this->GetNumberOfThreads());


	typedef itk::HalfHermitianToRealInverseFFTImageFilter< InternalComplexType,InternalType > IFFTFilterType;

	typename IFFTFilterType::Pointer ifft = IFFTFilterType::New();
	ifft->SetNumberOfThreads(this->GetNumberOfThreads());

	for(int i=0;i<6;i++){
		indexSelectionFilter->SetIndex(i);
		ifft->SetInput(indexSelectionFilter->GetOutput());
		ifft->Update();

		typename InternalType::Pointer ifftoutput =ifft->GetOutput();
		ifftoutput->DisconnectPipeline();

		composeResultFilter->SetInput(i,ifftoutput);


	}

	composeResultFilter->Update();

	this->SetPrimaryOutput(composeResultFilter->GetOutput());


}



#endif /* INCLUDE_TTTHESSIANIFFTIMAGEFILTER_HXX_ */
