#ifndef _TTTTENSORTOENERGYIMAGEFILTER_H_
#define _TTTTENSORTOENERGYIMAGEFILTER_H_
#include <itkImageToImageFilter.h>
#include <itkImage.h>
#include <itkUnaryFunctorImageFilter.h>
namespace ttt{
template<class TTensor,class TScalar> class EnergyFunctorType{
public:
	EnergyFunctorType(){}

  ~EnergyFunctorType() {};

  bool operator!=( const EnergyFunctorType & other ) const
     {
     return !(*this == other);
     }

   bool operator==( const EnergyFunctorType & other ) const
     {
     return true;
     }

   TScalar  operator()(const TTensor & a){

	   TScalar result =0;
	   double weights[]={1.0,2.0,2.0,1.0,2.0,1.0};
		for(int i=0;i<6;i++){
			result+=weights[i]*std::abs(a[i])*std::abs(a[i]);
		}
		return result;
	}
};

template<class ComplexHessianImageType,class ScalarImageType>
class TensorToEnergyImageFilter : public itk::UnaryFunctorImageFilter<ComplexHessianImageType,ScalarImageType,EnergyFunctorType<typename ComplexHessianImageType::PixelType,typename ScalarImageType::PixelType> > {
public:


	/** Standard class typedefs. */
  typedef TensorToEnergyImageFilter<ComplexHessianImageType,ScalarImageType>         Self;
  typedef itk::UnaryFunctorImageFilter<ComplexHessianImageType,ScalarImageType,EnergyFunctorType<typename ComplexHessianImageType::PixelType,typename ScalarImageType::PixelType> > Superclass;
  typedef itk::SmartPointer< Self >  Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorToEnergyImageFilter, ImageToImageFilter);
protected:
  TensorToEnergyImageFilter(){}
  ~TensorToEnergyImageFilter(){}


private:
  	  TensorToEnergyImageFilter(const Self &); //purposely not implemented
	void operator=(const Self &);  //purposely not implemented

};
}
#endif
