/*
 * tttDeconvolutionPoissonStageImageFilter.hxx
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_HXX_
#define INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_HXX_

#include "tttDeconvolutionPoissonStageImageFilter.h"

template<class TComplexImage, class TImage> ttt::DeconvolutionPoissonStageImageFilter<
		TComplexImage, TImage>::DeconvolutionPoissonStageImageFilter() {

	this->SetNumberOfRequiredInputs(4);
	this->SetNumberOfRequiredOutputs(2);

	this->SetNthOutput(0, this->MakeOutput(0));
	this->SetNthOutput(1, this->MakeOutput(1));

	//Z1 Pipeline

	// Set up minipipeline to compute estimate at each iteration
	m_ComplexMultiplyFilter1 = ComplexMultiplyType::New();
	m_ComplexMultiplyFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_ComplexMultiplyFilter1->ReleaseDataFlagOn();

	m_IFFTFilter1 = IFFTFilterType::New();
	m_IFFTFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	//m_IFFTFilter1->SetActualXDimensionIsOdd( this->GetXDimensionIsOdd() );
	m_IFFTFilter1->SetInput(m_ComplexMultiplyFilter1->GetOutput());
	m_IFFTFilter1->ReleaseDataFlagOn();

	m_AddFilter1 = AddType::New();
	m_AddFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_AddFilter1->SetInput1(m_IFFTFilter1->GetOutput());

	m_AddFilter1->ReleaseDataFlagOn();
	m_AddFilter1->Update();

	m_PoissonShrinkFilter1 = PoissonShrinkFilterType::New();
	m_PoissonShrinkFilter1->SetNumberOfThreads(this->GetNumberOfThreads());
	m_PoissonShrinkFilter1->SetInput1(m_AddFilter1->GetOutput());
	m_PoissonShrinkFilter1->GetFunctor().m_Alpha = this->m_Lambda;
	m_PoissonShrinkFilter1->ReleaseDataFlagOn();

	m_SubFilter1 = SubType::New();
	m_SubFilter1->SetInput1(m_PoissonShrinkFilter1->GetOutput());

	m_SubFilter1->ReleaseDataFlagOn();
	m_SubFilter1->SetNumberOfThreads(this->GetNumberOfThreads());

	m_FFTFilter2 = FFTFilterType::New();
	m_FFTFilter2->SetInput(m_SubFilter1->GetOutput());
	m_FFTFilter2->ReleaseDataFlagOn();
	m_FFTFilter2->SetNumberOfThreads(this->GetNumberOfThreads());

	m_ConjugateTransferFunction = ConjugateAdaptor::New();

	m_ComplexConjugateMultiplyFilter2 = ComplexConjugateMultiplyType::New();
	m_ComplexConjugateMultiplyFilter2->SetInput1(m_FFTFilter2->GetOutput());
	m_ComplexConjugateMultiplyFilter2->SetInput2(m_ConjugateTransferFunction);
	m_ComplexConjugateMultiplyFilter2->ReleaseDataFlagOn();
	m_ComplexConjugateMultiplyFilter2->SetNumberOfThreads(
			this->GetNumberOfThreads());

}

template<class TComplexImage, class TImage> void ttt::DeconvolutionPoissonStageImageFilter<
		TComplexImage, TImage>::GenerateData() {
	m_ComplexMultiplyFilter1->SetInput1(this->GetCurrentEstimation());
	m_ComplexMultiplyFilter1->SetInput2(this->GetTransferFunction());
	m_AddFilter1->SetInput2(this->GetLagrangeMultiplier());
	m_PoissonShrinkFilter1->SetInput2(this->GetInputImage());
	m_PoissonShrinkFilter1->Update();
	this->SetShrinkedImage(m_PoissonShrinkFilter1->GetOutput());

	m_SubFilter1->SetInput2(this->GetLagrangeMultiplier());
	m_ConjugateTransferFunction->SetImage(const_cast<TComplexImage*>(this->GetTransferFunction()));
	m_ComplexConjugateMultiplyFilter2->Update();

	this->SetConjugatedImage(m_ComplexConjugateMultiplyFilter2->GetOutput());
}

#endif /* INCLUDE_TTTDECONVOLUTIONPOISSONSTAGEIMAGEFILTER_HXX_ */
