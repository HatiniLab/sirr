/*
 * tttDeconvolutionBoundsShrinkImageFilter.hxx
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_HXX_
#define INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_HXX_
#include "tttDeconvolutionBoundsStageImageFilter.h"
template<class TImage,class TComplexImage> ttt::DeconvolutionBoundsStageImageFilter<
TImage, TComplexImage>::DeconvolutionBoundsStageImageFilter() {

	this->SetNumberOfRequiredInputs(2);
	this->SetNumberOfRequiredOutputs(2);

	this->SetNthOutput(0, this->MakeOutput(0));
	this->SetNthOutput(1, this->MakeOutput(1));

	m_AddFilter2 = AddType::New();
	m_AddFilter2->SetNumberOfThreads(this->GetNumberOfThreads());

	m_BoundsShrinkFilter1 = BoundsShrinkFilterType::New();
	m_BoundsShrinkFilter1->SetInput(m_AddFilter2->GetOutput());
	m_BoundsShrinkFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_BoundsShrinkFilter1->ReleaseDataFlagOn();
	m_BoundsShrinkFilter1->InPlaceOn();

	m_SubFilter2 = SubType::New();
	m_SubFilter2->SetInput1(m_BoundsShrinkFilter1->GetOutput());
	m_SubFilter2->ReleaseDataFlagOn();
	m_SubFilter2->SetNumberOfThreads(this->GetNumberOfThreads());

	m_FFTFilter3 = FFTFilterType::New();
	m_FFTFilter3->SetInput(m_SubFilter2->GetOutput());
	m_FFTFilter3->ReleaseDataFlagOn();
	m_FFTFilter3->SetNumberOfThreads(this->GetNumberOfThreads());

};

template<class TImage,class TComplexImage> void ttt::DeconvolutionBoundsStageImageFilter<
TImage, TComplexImage>::GenerateData() {
	m_AddFilter2->SetInput1(this->GetCurrentEstimation());
	m_AddFilter2->SetInput2(this->GetLagrangeMultiplier());
	m_BoundsShrinkFilter1->Update();
	this->SetShrinkedImage(m_BoundsShrinkFilter1->GetOutput());


	m_SubFilter2->SetInput2(this->GetLagrangeMultiplier());
	m_FFTFilter3->Update();

	this->SetConjugatedImage(m_FFTFilter3->GetOutput());
}


#endif /* INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_HXX_ */
