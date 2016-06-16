/*
 * tttBoundsShrink.h
 *
 *  Created on: Nov 19, 2014
 *      Author: morgan
 */

#ifndef INCLUDE_TTTBOUNDSSHRINKIMAGEFILTER_H_
#define INCLUDE_TTTBOUNDSSHRINKIMAGEFILTER_H_
#include <itkUnaryFunctorImageFilter.h>
namespace ttt{
template< class TReal>
class BoundsShrinkFunctor{
public:

	BoundsShrinkFunctor(){

	}
	bool operator!=( const BoundsShrinkFunctor & other ) const
		{
	    return !(*this == other);
	    }
	bool operator==( const BoundsShrinkFunctor & other ) const
		{
	    return true;
	    }
	TReal operator()( TReal  value)
	{
		return std::max((TReal)0.0,value);
	}
};
template<class TImage>
class BoundsShrinkImageFilter : public itk::UnaryFunctorImageFilter<TImage,TImage,BoundsShrinkFunctor<typename TImage::PixelType> >{
public:
	/** Standard class typedefs. */
	typedef BoundsShrinkImageFilter<TImage>         Self;
	  typedef itk::UnaryFunctorImageFilter<TImage,TImage,BoundsShrinkFunctor<typename TImage::PixelType> > Superclass;
	  typedef itk::SmartPointer< Self >  Pointer;

	  /** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Run-time type information (and related methods). */
	  itkTypeMacro(BoundsShrinkImageFilter, UnaryFunctorImageFilter);

protected:
	  BoundsShrinkImageFilter(){}
  ~BoundsShrinkImageFilter(){}


private:
  BoundsShrinkImageFilter(const Self &); //purposely not implemented
	void operator=(const Self &);  //purposely not implemented

};

}

#endif /* INCLUDE_TTTBOUNDSSHRINKIMAGEFILTER_H_ */
