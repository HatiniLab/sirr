/*
 * tttDeconvolutionHessianStageImageFilter.h
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_H_
#define INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_H_
#include <itkImageToImageFilter.h>
#include <tttMultiplyImageByTensorImageFilter.h>
#include <tttHessianIFFTImageFilter.h>
#include <tttHessianFFTImageFilter.h>
#include <tttHessianShrinkImageFilter.h>
#include <tttMultiplyTensorByConjugateTensorImageFilter.h>
#include <tttReduceTensorImageFilter.h>
namespace ttt {

template<class TComplexImage, class TComplexHessian, class THessian> class DeconvolutionHessianStageImageFilter: public itk::ImageToImageFilter<
		TComplexImage, THessian> {
public:
	typedef DeconvolutionHessianStageImageFilter Self;
	typedef itk::ImageToImageFilter<TComplexImage,THessian> Superclass;
	typedef itk::SmartPointer<Self> Pointer;


	itkNewMacro(Self);
	itkTypeMacro(Self,Superclass);

	virtual void SetCurrentEstimation(const  TComplexImage * image) {
		this->SetNthInput(0, const_cast<TComplexImage*>(image));
	}
	virtual typename TComplexImage::ConstPointer GetCurrentEstimation() {
		return static_cast<const TComplexImage *>(this->itk::ProcessObject::GetInput(
				0));
	}

	virtual void SetHessianFilter(const TComplexHessian* hessian) {
		this->SetNthInput(1, const_cast<TComplexHessian*>(hessian));
	}

	virtual const TComplexHessian* GetHessianFilter() {
		return static_cast<const TComplexHessian *>(this->itk::ProcessObject::GetInput(
				1));
	}

	virtual void SetLagrangeMultiplier(const THessian* transfer) {
		this->SetNthInput(2, const_cast<THessian*>(transfer));
	}
	virtual typename THessian::ConstPointer GetLagrangeMultiplier() {
		return static_cast<const THessian *>(this->itk::ProcessObject::GetInput(2));
	}

	virtual typename THessian::Pointer GetShrinkedHessian() {
		return static_cast<THessian *>(this->GetOutput(0));
	}
	virtual typename TComplexImage::Pointer GetConjugatedHessian() {
		return dynamic_cast<TComplexImage *>(this->GetOutput(1));
	}


protected:
	DeconvolutionHessianStageImageFilter();

	~DeconvolutionHessianStageImageFilter() {

	}

	itk::DataObject::Pointer MakeOutput(unsigned int idx) {
		itk::DataObject::Pointer output;

		switch (idx) {
		case 0:
			output = (THessian::New()).GetPointer();
			break;
		case 1:
			output = (TComplexImage::New()).GetPointer();
			break;
		default:
			std::cerr << "No output " << idx << std::endl;
			output = NULL;
			break;
		}
		return output.GetPointer();
	}
	virtual void SetShrinkedHessian(THessian* shrinked) {
		this->SetNthOutput(0, static_cast<THessian*>(shrinked));
	}

	virtual void SetConjugatedHessian(TComplexImage* conjugated) {
		this->SetNthOutput(1, static_cast<TComplexImage*>(conjugated));
	}

	virtual void GenerateData();
private:
	typedef ttt::MultiplyByTensorImageFilter<TComplexImage,
			TComplexHessian> HessianFrequencyFilterType;
	typedef ttt::HessianIFFTImageFilter<TComplexHessian,
			THessian> HessianIFFTFilterType;

	typedef ttt::HessianFFTImageFilter<THessian, TComplexHessian> HessianFFTFilterType;

	typedef itk::AddImageFilter<THessian> AddHessianType;
	typedef itk::SubtractImageFilter<THessian> SubHessianType;

	typedef ttt::HessianShrinkImageFilter<THessian> HessianShrinkFilterType;
	typedef ttt::MultiplyTensorByConjugateTensorImageFilter<
			TComplexHessian> ConjugateHessianFrequencyFilterType;

	typedef ttt::ReduceTensorImageFilter<TComplexHessian,
			TComplexImage> HessianReducerFilterType;

	typename HessianFrequencyFilterType::Pointer m_HessianFrequencyEstimationFilter1;

	typename HessianIFFTFilterType::Pointer m_HessianIFFTFilter1;
	typename AddHessianType::Pointer m_AddHessianFilter1;

	typename HessianShrinkFilterType::Pointer m_HessianShrinkFilter1;

	typename SubHessianType::Pointer m_SubHessianFilter1;
	typename HessianFFTFilterType::Pointer m_HessianFFTFilter1;

	typename ConjugateHessianFrequencyFilterType::Pointer m_ConjugateHessianFilter1;

	typename HessianReducerFilterType::Pointer m_HessianReduceFilter1;

};
#include "tttDeconvolutionHessianStageImageFilter.hxx"

}
#endif /* INCLUDE_TTTDECONVOLUTIONHESSIANSTAGEIMAGEFILTER_H_ */
