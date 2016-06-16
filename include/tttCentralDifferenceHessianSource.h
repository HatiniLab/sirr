/*
 * tttCentralDifferenceHessianSource.h
 *
 *  Created on: Nov 14, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_H_
#define INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_H_

#include <itkImageSource.h>
#include <itkImage.h>
#include <itkSymmetricSecondRankTensor.h>
namespace ttt{

template<class TReal ,int dim > class CentralDifferenceHessianSource : public itk::ImageSource< itk::Image<itk::SymmetricSecondRankTensor<std::complex<TReal>,dim > ,dim > > {
public:
	typedef itk::Image< itk::SymmetricSecondRankTensor<std::complex<TReal>,dim > ,dim > HessianFilterImageType;
	typedef itk::Image< std::complex<TReal> ,dim > FilterImageType;

	typedef itk::Image< TReal,dim > EnergyImageType;
	/** Standard class typedefs. */
  typedef CentralDifferenceHessianSource<TReal,dim>         Self;
  typedef itk::ImageSource<HessianFilterImageType >  Superclass;
  typedef itk::SmartPointer< Self >  Pointer;

  typedef itk::Image<TReal,dim> KernelImageType;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CentralDifferenceHessianSource, ImageSource);

  itkGetMacro(LowerPad,typename KernelImageType::SizeType);
  itkSetMacro(LowerPad,typename KernelImageType::SizeType);

  itkGetMacro(PadSize,typename KernelImageType::SizeType);
  itkSetMacro(PadSize,typename KernelImageType::SizeType);

  itkGetMacro(Spacing,typename KernelImageType::SpacingType);
  itkSetMacro(Spacing,typename KernelImageType::SpacingType);

protected:
  CentralDifferenceHessianSource(){
  }
  ~CentralDifferenceHessianSource(){}

  /** Does the real work. */
  virtual void GenerateData();
  virtual void AllocateOutputs();

private:
  CentralDifferenceHessianSource(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

  void InitHessianXX();
  void InitHessianXY();
  void InitHessianXZ();
  void InitHessianYY();
  void InitHessianYZ();
  void InitHessianZZ();
  void HessianKernelToHessianOperator(const typename KernelImageType::Pointer & kernel, typename FilterImageType::Pointer & op);

  typename KernelImageType::SizeType m_LowerPad;
  typename KernelImageType::SizeType m_PadSize;
  typename KernelImageType::SpacingType m_Spacing;


  std::vector<typename FilterImageType::Pointer> m_Filters;


};


}

#include "tttCentralDifferenceHessianSource.hxx"

#endif /* INCLUDE_TTTCENTRALDIFFERENCEHESSIANSOURCE_H_ */
