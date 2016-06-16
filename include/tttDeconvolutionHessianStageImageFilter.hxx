/*
 * tttDeconvolutionHessianStageImageFilter.hxx
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_HXX_
#define INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_HXX_

#include "tttDeconvolutionHessianStageImageFilter.h"

template<class TComplexImage, class TComplexHessian, class THessian> ttt::DeconvolutionHessianStageImageFilter<
		TComplexImage, TComplexHessian, THessian>::DeconvolutionHessianStageImageFilter() {
	this->SetNumberOfRequiredInputs(3);
	this->SetNumberOfRequiredOutputs(2);

	this->SetNthOutput(0, this->MakeOutput(0));
	this->SetNthOutput(1, this->MakeOutput(1));

	m_HessianFrequencyEstimationFilter1 = HessianFrequencyFilterType::New();
	m_HessianFrequencyEstimationFilter1->SetNumberOfThreads(
			this->GetNumberOfThreads());

	m_HessianIFFTFilter1 = HessianIFFTFilterType::New();
	m_HessianIFFTFilter1->SetInput(
			m_HessianFrequencyEstimationFilter1->GetOutput());
	m_HessianIFFTFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_AddHessianFilter1 = AddHessianType::New();
	m_AddHessianFilter1->SetInput1(m_HessianIFFTFilter1->GetOutput());

	//m_AddHessianFilter1->InPlaceOn();
	m_AddHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_AddHessianFilter1->ReleaseDataFlagOn();

	m_HessianShrinkFilter1 = HessianShrinkFilterType::New();
	m_HessianShrinkFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_HessianShrinkFilter1->SetInput(m_AddHessianFilter1->GetOutput());
	m_HessianShrinkFilter1->GetFunctor().m_Lambda = 1; //m_Lambda;
	m_HessianShrinkFilter1->InPlaceOn();
	m_HessianShrinkFilter1->ReleaseDataFlagOn();

	m_SubHessianFilter1 = SubHessianType::New();
	m_SubHessianFilter1->SetInput1(m_HessianShrinkFilter1->GetOutput());

	m_SubHessianFilter1->ReleaseDataFlagOn();
	m_SubHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_HessianFFTFilter1 = HessianFFTFilterType::New();
	m_HessianFFTFilter1->SetInput(m_SubHessianFilter1->GetOutput());
	m_HessianFFTFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_ConjugateHessianFilter1 = ConjugateHessianFrequencyFilterType::New();
	m_ConjugateHessianFilter1->SetInput1(m_HessianFFTFilter1->GetOutput());

	m_ConjugateHessianFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_HessianReduceFilter1 = HessianReducerFilterType::New();
	m_HessianReduceFilter1->SetInput(m_ConjugateHessianFilter1->GetOutput());
	m_HessianReduceFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

}

template<class TComplexImage, class TComplexHessian, class THessian> void ttt::DeconvolutionHessianStageImageFilter<
		TComplexImage, TComplexHessian, THessian>::GenerateData() {
	m_HessianFrequencyEstimationFilter1->SetInput1(
			this->GetCurrentEstimation());
	m_HessianFrequencyEstimationFilter1->SetInput2(this->GetHessianFilter());

	m_AddHessianFilter1->SetInput2(this->GetLagrangeMultiplier());
	m_HessianShrinkFilter1->Update();
	this->SetShrinkedHessian(m_HessianShrinkFilter1->GetOutput());
	m_SubHessianFilter1->SetInput2(this->GetLagrangeMultiplier());

	m_ConjugateHessianFilter1->SetInput2(this->GetHessianFilter());

	m_HessianReduceFilter1->Update();
	this->SetConjugatedHessian(m_HessianReduceFilter1->GetOutput());
}

#endif /* INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_HXX_ */
