/*
 * tttDeconvolutionBoundsStageImageFilter.h
 *
 *  Created on: Nov 21, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_H_
#define INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_H_
#include "tttBoundsShrinkImageFilter.h"
namespace ttt {
template<class TImage, class TComplexImage> class DeconvolutionBoundsStageImageFilter: public itk::ImageToImageFilter<
	TImage, TComplexImage> {
public:

	typedef DeconvolutionBoundsStageImageFilter<TImage,TComplexImage> Self;
	typedef itk::ImageToImageFilter<TImage,TComplexImage> Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	itkNewMacro(Self);
	itkTypeMacro(Self,Superclass);
	typedef TImage ImageType;
	typedef TComplexImage ComplexImageType;

	virtual void SetCurrentEstimation(const TImage*  image) {
		this->SetNthInput(0, const_cast<TImage*>(image));
	}
	virtual typename TImage::ConstPointer GetCurrentEstimation() {
		return static_cast<const TImage *>(this->ProcessObject::GetInput(
				0));
	}
	virtual void SetLagrangeMultiplier(const TImage* transfer) {
		this->SetNthInput(1, const_cast<TImage*>(transfer));
	}
	virtual typename TImage::ConstPointer GetLagrangeMultiplier() {
		return static_cast<const TImage *>(this->ProcessObject::GetInput(1));
	}


	virtual typename TImage::Pointer GetShrinkedImage() {
			return dynamic_cast<TImage *>(this->GetOutput(0));
	}
	virtual typename TComplexImage::Pointer GetConjugatedImage() {
			return static_cast<TComplexImage *>(this->GetOutput(1));
	}
protected:
	DeconvolutionBoundsStageImageFilter();
	~DeconvolutionBoundsStageImageFilter(){

	}

	virtual void SetShrinkedImage(TImage*  shrinked) {
		this->SetNthOutput(0, static_cast<TImage*>(shrinked));
	}

	virtual void SetConjugatedImage(
			TComplexImage* deconvoluted) {
		this->SetNthOutput(1, static_cast<TComplexImage*>(deconvoluted));
	}

	virtual void GenerateData();

private:
	typedef itk::AddImageFilter<ImageType> AddType;
	typedef itk::SubtractImageFilter<ImageType> SubType;
	typedef ttt::BoundsShrinkImageFilter<ImageType> BoundsShrinkFilterType;
	typedef  itk::ForwardFFTImageFilter<ImageType,ComplexImageType> FFTFilterType;

	typename AddType::Pointer						m_AddFilter2;
	typename BoundsShrinkFilterType::Pointer		m_BoundsShrinkFilter1;
	typename SubType::Pointer						m_SubFilter2;
	typename FFTFilterType::Pointer				m_FFTFilter3;

};
}

#include "tttDeconvolutionBoundsStageImageFilter.hxx"



#endif /* INCLUDE_TTTDECONVOLUTIONBOUNDSSTAGEIMAGEFILTER_H_ */
