package com.shimon_doodkin.imshow;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;


 
public class Algorithm {

  public static abstract class ImShowMethod 
	{ 
		void show_gray(final String name,final Mat m)
		{
			imshow(name, m,Imgproc.COLOR_GRAY2RGBA, 4);
		}
		
		void show_bgr(final String name,final Mat m)
		{
			imshow(name, m,Imgproc.COLOR_RGB2BGRA, 0);
		}
		
		void imshow(final String name,final Mat m,int cvtColor_convert_code)
		{
			this.imshow(name,m,cvtColor_convert_code,0);
		}

		void imshow(final String name,final Mat m,int cvtColor_convert_code,int cvtColor_dest_cdn)
		{
			
			Mat tmp = new Mat ( m.rows(), m.cols(), CvType.CV_8U, new Scalar(4));
			// Imgproc.cvtColor(m, tmp, Imgproc.COLOR_RGB2BGRA);
			// Imgproc.cvtColor(m, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
			Imgproc.cvtColor(m, tmp, cvtColor_convert_code, cvtColor_dest_cdn);
			imshow(name,tmp);			
		}
		
		abstract void imshow(final String name,final Mat rgba_mat);
	}
 
	  public static ImShowMethod imshow=null;
	  
	  public static void main(String[] args) {
	  
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		// reflection is in order for this Algorithm class to compile natively from an android project and run on pc with run settings
		imshow=new ImShowMethod() {
			  @SuppressWarnings("unused")
			  class ReflectionClass // some utility reflection tool class
			  {
				  Object object;
				  String name;
				  Class<?> classForName;

				  public ReflectionClass(String name) //construct
				  {
					  this.name=name;
					  try {
						  classForName=Class.forName(this.name);
						  object = classForName.getDeclaredConstructor().newInstance();
					  } catch (Exception e) {e.printStackTrace();}
				  }
				  
				  public ReflectionClass(String name, Class<?> [] search_constructor_with_types,Object... values) //construct
				  {
					  this.name=name;
					  try {
						  classForName=Class.forName(this.name);
						  object = classForName.getDeclaredConstructor(search_constructor_with_types).newInstance(values);
					  } catch (Exception e) {e.printStackTrace();}
				  }
				  
				  public ReflectionClass(String name, Object value) //cast
				  {
					  this.name=name;

					  try {
						  classForName=Class.forName(this.name);
						  this.object=classForName.cast(value);
					  } catch (Exception e) {e.printStackTrace();}
				  }
				  
				  public Object method(String name, Class<?> [] search_method_with_types,Object[] values)
				  {
					try {
						return classForName.getMethod (name, search_method_with_types).invoke (object, values );
					} catch (Exception e) {e.printStackTrace();}
					return null;
				  }

				  
				  public Object method(String name)
				  {
					try {
						return classForName.getMethod (name).invoke (object);
					} catch (Exception e) {e.printStackTrace();}
					return null;
				  }

				  public ReflectionClass wrap(String wrapclassname,Object value)
				  {
					try {
						return new ReflectionClass(wrapclassname,  value );
					} catch (Exception e) {e.printStackTrace();}
					return null;
				  }
				  
				  public ReflectionClass methodWithWrap(String name, Class<?> [] types, Object[] values,String wrapclassname)
				  {
					  return wrap(wrapclassname,  method(name,types,values) );
				  }

				  
				  public ReflectionClass methodWithWrap(String name,String wrapclassname)
				  {
					  return wrap(wrapclassname,  method(name) );
				  }
				  
				  public ReflectionClass cast(String wrapclassname)
				  {
					  return wrap(wrapclassname,  object );
				  }
				  
				  public Object getField(String name)
				  {
					try {
						return classForName.getDeclaredField (name).get(object);
					} catch (Exception e) {e.printStackTrace();}
					return null;
				  }
				  
				  public ReflectionClass getFieldWithWrap(String name,String wrapclassname)
				  {
				   return wrap(wrapclassname,  getField(name) );
				  }
				  
				  public void setField(String name,Object value)
				  {
					try {
						classForName.getDeclaredField(name).set(object,value);
					} catch (Exception e) {e.printStackTrace();}
				  }
			  }
			  
			@SuppressWarnings("unused")
			private ReflectionClass toBufferedImage(Mat BGRA_mat) {
				
				final int	TYPE_3BYTE_BGR	=5; // from https://docs.oracle.com/javase/7/docs/api/constant-values.html#java.awt.image.BufferedImage.TYPE_4BYTE_ABGR
				final int	TYPE_4BYTE_ABGR=	6;
				final int	TYPE_4BYTE_ABGR_PRE	=7;
				final int	TYPE_BYTE_BINARY	=12;
				final int	TYPE_BYTE_GRAY	=10;
				final int	TYPE_BYTE_INDEXED	=13;
				final int	TYPE_CUSTOM	=0;
				final int	TYPE_INT_ARGB	=2;
				final int	TYPE_INT_ARGB_PRE	=3;
				final int	TYPE_INT_BGR	=4;
				final int	TYPE_INT_RGB	=1;
				final int	TYPE_USHORT_555_RGB=	9;
				final int	TYPE_USHORT_565_RGB=	8;
				final int	TYPE_USHORT_GRAY	=11;
				
				
				int bufferSize = BGRA_mat.channels() * BGRA_mat.cols() * BGRA_mat.rows();
				byte[] inputPixels_BGRA = new byte[bufferSize];
				BGRA_mat.get(0, 0, inputPixels_BGRA); // get all the pixels

				ReflectionClass image=new ReflectionClass("java.awt.image.BufferedImage", new Class[]{int.class,int.class,int.class} ,  BGRA_mat.cols(), BGRA_mat.rows(), TYPE_4BYTE_ABGR);
				
				ReflectionClass rastar=new ReflectionClass("java.awt.image.Raster",  image.method("getRaster") );
				
				ReflectionClass databuffer=new ReflectionClass("java.awt.image.DataBuffer",  rastar.method("getDataBuffer") );
				ReflectionClass databufferbyte=new ReflectionClass("java.awt.image.DataBufferByte",  databuffer.object );

				final byte[] targetPixels_ARGB = (byte[]) databufferbyte.method("getData");
				
				//System.arraycopy(b, 0, targetPixels, 0, b.length);
				
				int count= inputPixels_BGRA.length,pixel=0;
				while(count-->0){
				       targetPixels_ARGB[pixel]   = inputPixels_BGRA[pixel+3];
				       targetPixels_ARGB[pixel+1] = inputPixels_BGRA[pixel+2];
				       targetPixels_ARGB[pixel+2] = inputPixels_BGRA[pixel+1];
				       targetPixels_ARGB[pixel+3] = inputPixels_BGRA[pixel];
				       pixel+=4;
				       count-=4;
				   }
				return image;
			}
			
			@Override
			public void imshow(String name, Mat BGRA_mat) {		
				try {
					ReflectionClass bufferdimage=toBufferedImage(BGRA_mat);
					ReflectionClass frame=new ReflectionClass("javax.swing.JFrame"); 
					ReflectionClass imageicon=new ReflectionClass("javax.swing.ImageIcon", new Class[]{ Class.forName("java.awt.Image")  }, bufferdimage.object );
					ReflectionClass label=new ReflectionClass("javax.swing.JLabel", new Class[]{ Class.forName("javax.swing.Icon")    }, imageicon.object );
					
					ReflectionClass container=frame.methodWithWrap("getContentPane","java.awt.Container");
					container.method("add", new Class[]{  Class.forName("java.awt.Component")  },new Object[] {label.object} );
					frame.method("pack");
					frame.method("setVisible", new Class[]{ boolean.class  },new Object[] {true} );
			    } catch (Exception e) {
			        e.printStackTrace();
			    }
			}
		};
		
		
		
		Mat image = Highgui.imread("C:/Users/USER/Downloads/static1.jpg");   // Read the file // put some file you have here
	  imshow.imshow( "Display window", image ,Imgproc.COLOR_RGB2BGRA);     // Show our image inside it.
 

	  }
}
