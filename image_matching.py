import numpy as np 
import cv2 


#to match Template 
def match_templplate(main_img,find_img):
    #method=['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED']
    try:
        result=cv2.matchTemplate(main_img,find_img,cv2.TM_CCORR_NORMED)
        print(result)
    except:
        return None
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)
    top_left=max_loc
    h,w=find_img.shape[:2]
    bottom_right=((top_left[0]+w),(top_left[1]+h))    
    cv2.rectangle(main_img,top_left,bottom_right,(255,0,0))
    return main_img

#Feature selection 
def sift_image(main_img,second_img,name):
    sift = cv2.SIFT.create()
    
    kp1,dest1=sift.detectAndCompute(main_img,None)
    kp2,dest2=sift.detectAndCompute(second_img,None)
    bf=cv2.BFMatcher()
    match=bf.knnMatch(dest1,dest2,k=2)
    good_match=[]
    for match1,match2 in match:
        if match1.distance < 0.5*match2.distance:
            good_match.append([match1])
    print(len(good_match))
    return len(good_match)
    
    
    

#img database
image_name_list = ['images\Daffodil.jfif','images\Daisy.jfif','images/Hydrangea.jfif',
                   'images\Lily.jfif','images\Marigold.jfif','images/Orchid.jfif',
                   'images\Rose.webp','images\Sunflower.jfif','images\Tulip.jfif'                   
                   ]
image_numpy_list=[]

#loading of images
for i in image_name_list:
    try:
        z=cv2.imread(r""+i)
        image_numpy_list.append(z)
    except Exception as e:
        pass
    



main_image=None
#reading the subimage
crop_image=cv2.imread('images/Crop_Tulip.jpg')

match_list=[]

#calling feature selection for all image
for i in range(len(image_numpy_list)):

    value=sift_image(image_numpy_list[i],crop_image,image_name_list[i])
    match_list.append(value)
max_1=max(match_list)
index=match_list.index(max_1)
print(index)

#matching the template
main_image = match_templplate(image_numpy_list[index],crop_image)

#displaying the  image 

cv2.imshow("main_image",main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

