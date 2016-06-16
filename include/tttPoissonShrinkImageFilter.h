/*
 * tttPoissonShrink.h
 *
 *  Created on: Nov 19, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTPOISSONSHRINKIMAGEFILTER_H_
#define INCLUDE_TTTPOISSONSHRINKIMAGEFILTER_H_
#include <itkBinaryFunctorImageFilter.h>
namespace ttt{

template< class TReal>
class PoissonShrinkFunctor
{
public:
	PoissonShrinkFunctor()
   {
   m_Alpha = itk::NumericTraits< TReal >::min();
   };
  ~PoissonShrinkFunctor() {};
  bool operator!=( const PoissonShrinkFunctor & other ) const
    {
    return !(*this == other);
    }
  bool operator==( const PoissonShrinkFunctor & other ) const
    {
    return true;
    }
  inline TReal operator()( const TReal & u,const TReal & y)

      {
	  TReal shrinked=(u - 1.0/m_Alpha + vcl_sqrt(vnl_math_sqr(u - 1/m_Alpha) +4*y/m_Alpha))/2;


	  return shrinked;
    }
  TReal m_Alpha;
};

template<class TImage>
class PoissonShrinkImageFilter : public itk::BinaryFunctorImageFilter<TImage,TImage,TImage,PoissonShrinkFunctor<typename TImage::PixelType> >{
public:
	/** Standard class typedefs. */
  typedef PoissonShrinkImageFilter<TImage>         Self;
  typedef itk::BinaryFunctorImageFilter<TImage,TImage,TImage,PoissonShrinkFunctor<typename TImage::PixelType> > Superclass;
  typedef itk::SmartPointer< Self >  Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PoissonShrinkImageFilter, BinaryFunctorImageFilter);
protected:
  PoissonShrinkImageFilter(){}
  ~PoissonShrinkImageFilter(){}


private:
  PoissonShrinkImageFilter(const Self &); //purposely not implemented
	void operator=(const Self &);  //purposely not implemented
};
}
#endif /* INCLUDE_TTTPOISSONSHRINKIMAGEFILTER_H_ */
