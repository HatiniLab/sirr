/*
 * tttHessianFFTImageFilter.hxx
 *
 *  Created on: Oct 23, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTHESSIANFFTIMAGEFILTER_HXX_
#define INCLUDE_TTTHESSIANFFTIMAGEFILTER_HXX_

//#include <itkConstNthElementImageAdaptor.h>
#include "itkNthElementImageAdaptor.h"
#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#include <itkImageDuplicator.h>
#include "tttHessianFFTImageFilter.h"
#include "itkComposeImageFilter.h"

#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkPasteImageFilter.h"

template< class TInputImage, class TOutputImage >
void
ttt::HessianFFTImageFilter< TInputImage, TOutputImage >
::GenerateOutputInformation()
{
// Get pointers to the input and output.
typename TInputImage::ConstPointer inputPtr = this->GetInput();
typename TOutputImage::Pointer outputPtr = this->GetOutput();
if ( !inputPtr || !outputPtr )
{
return;
}
// This is all based on the same function in itk::ShrinkImageFilter
// ShrinkImageFilter also modifies the image spacing, but spacing
// has no meaning in the result of an FFT.
const typename TInputImage::SizeType inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
const typename TInputImage::IndexType inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();
typename TOutputImage::SizeType outputSize;
typename TOutputImage::IndexType outputStartIndex;
// In 4.3.4 of the FFTW documentation, they indicate the size of
// of a real-to-complex FFT is N * N ... + (N /2+1)
// 1 2 d
// complex numbers.
// static_cast probably not necessary but want to make sure integer
// division is used.
outputSize[0] = static_cast< unsigned int >( inputSize[0] ) / 2 + 1;
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
void
ttt::HessianFFTImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
{
// Call the superclass implementation of this method.
Superclass::GenerateInputRequestedRegion();
// Get pointer to the input.
typename TInputImage::Pointer input =
const_cast< TInputImage * >( this->GetInput() );
if ( !input )
{
return;
}
input->SetRequestedRegionToLargestPossibleRegion();
}
template< class TInputImage, class TOutputImage >
void
ttt::HessianFFTImageFilter< TInputImage, TOutputImage >
::EnlargeOutputRequestedRegion(DataObject *output)
{
Superclass::EnlargeOutputRequestedRegion(output);
output->SetRequestedRegionToLargestPossibleRegion();
}



template<class TInputImage,class TOutputImage>
void ttt::HessianFFTImageFilter<TInputImage,TOutputImage>::GenerateData(){
	//this->AllocateOutputs();

	typedef itk::Image<double,TInputImage::ImageDimension> InternalType;
	typedef itk::Image<std::complex<double>,TInputImage::ImageDimension> InternalComplexType;

	typedef itk::VectorIndexSelectionCastImageFilter<TInputImage, InternalType> IndexSelectionType;
	typename IndexSelectionType::Pointer indexSelectionFilter = IndexSelectionType::New();

	typedef itk::ComposeImageFilter<InternalComplexType, TOutputImage> ComposeCovariantVectorImageFilterType;




	indexSelectionFilter->SetInput(this->GetInput());
	indexSelectionFilter->SetNumberOfThreads(this->GetNumberOfThreads());

	typename ComposeCovariantVectorImageFilterType::Pointer composeResultFilter = ComposeCovariantVectorImageFilterType::New();
	composeResultFilter->SetNumberOfThreads(this->GetNumberOfThreads());

	for(unsigned int i=0;i<6;i++){
		indexSelectionFilter->SetIndex(i);

		typedef itk::RealToHalfHermitianForwardFFTImageFilter< InternalType,InternalComplexType > FFTFilterType;
		typename FFTFilterType::Pointer fft = FFTFilterType::New();
		indexSelectionFilter->Update();


		fft->SetInput(indexSelectionFilter->GetOutput());
		fft->SetNumberOfThreads(this->GetNumberOfThreads());
		fft->Update();
		typename InternalComplexType::Pointer fftoutput =fft->GetOutput();
		fftoutput->DisconnectPipeline();

		composeResultFilter->SetInput(i,fftoutput);
	}

	composeResultFilter->Update();

	this->SetPrimaryOutput(composeResultFilter->GetOutput());
	//itk::ImageAlgorithm::Copy(composeResultFilter->GetOutput(), this->GetOutput(), composeResultFilter->GetOutput()->GetRequestedRegion(),	                       this->GetOutput()->GetRequestedRegion() );


}



#endif /* INCLUDE_TTTHESSIANFFTIMAGEFILTER_HXX_ */
