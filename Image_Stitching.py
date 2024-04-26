
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import sys
import itertools


def fundamental_matrix(kp1, kp2, good_matches):
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            F, F_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            return F, F_mask, pts1, pts2


def ratio_test(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def crop_img(img1, img2, H):
    h1,w1,_ = img1.shape
    h2,w2,_ = img2.shape
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    corners1 = corners1.reshape(-1, 1, 2) 
    corners_transformed = cv2.perspectiveTransform(corners1, H)
    all_corners = np.concatenate((corners1, corners_transformed), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    T = np.array([[1, 0, -1*xmin], [0, 1, -1*ymin], [0, 0, 1]], dtype=np.float32)
    mosaic1 = cv2.warpPerspective(img1, T @ H, (xmax-xmin, ymax-ymin))
    mosaic2 = cv2.warpPerspective(img2, T, (xmax-xmin, ymax-ymin))
    if mosaic1.shape[0] > 2000:
        scaling_factor = mosaic1.shape[0] // 2000
        mosaic1 = cv2.resize(mosaic1, (mosaic1.shape[1]//scaling_factor, mosaic1.shape[0]//scaling_factor))
        mosaic2 = cv2.resize(mosaic2, (mosaic2.shape[1]//scaling_factor, mosaic2.shape[0]//scaling_factor))

    return mosaic1, mosaic2

def display_img(mosaic_display_img):
    resized_img = cv2.resize(mosaic_display_img, (mosaic_display_img.shape[1]//2, mosaic_display_img.shape[0]//2))
    cv2.imshow('Mosaic', resized_img)
    key = cv2.waitKey(0)
    if key in [27, ord('q'), ord('Q')]:
        cv2.destroyAllWindows()



def Laplacian_pyramid_blend(A, B):
    #blend using Laplacian pyramid method

    # Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpA[i])
        gpA[i - 1] = cv2.resize(gpA[i - 1], (GE.shape[1], GE.shape[0]))
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpB[i])
        gpB[i - 1] = cv2.resize(gpB[i - 1], (GE.shape[1], GE.shape[0]))
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
    # add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols, _= la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)
    # reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def get_images(img_dir):
    start_cwd = os.getcwd()
    os.chdir(img_dir)
    names = os.listdir('./')
    names = [name for name in names if 'jpg' in name.lower()]
    names.sort()


    images = []
    for n in names:
        im = cv2.imread(n).astype('uint8')
        if im is None:
            print('Could not open', n)
            sys.exit(0)
        if im.shape[0] > 1000:
            im = cv2.resize(im, (im.shape[1]//5, im.shape[0]//5))
        images.append(im)

    os.chdir(start_cwd)
    names = [os.path.splitext(name)[0] for name in names]
    return names, images



def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2



def img_mosaic(i, j, image1, image2, names, imgs, sift, bf, directory, is_mosaic=False):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype('uint8') 
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype('uint8') 

    # Step 1: Extract keypoints and descriptors
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # Step 2: Match keypoints and descriptors
    matches = bf.knnMatch(desc1, desc2, k=2) 

    # ratio test
    good_matches = ratio_test(matches)

    # save keypoints
    if not is_mosaic:
        img_matches = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        input_files = [names[i], names[j]]
        output_filename = ("_".join(sorted(input_files))) + "_sift.jpg"
        output_path = os.path.join(directory+"/", output_filename)
        cv2.imwrite(output_path, img_matches)

        print(f"SIFT matching results after ratio test: {len(good_matches)}")
        print(f"Fraction of matches for img{i}: {len(good_matches)/ len(kp1)}")
        print(f"Fraction of matches for img{j}: {len(good_matches)/ len(kp2)}")
        print()

    # Step 3: Decision based on the number and fraction of matches
    if len(good_matches)/len(kp1) + len(good_matches)/len(kp2) <= 0.05:
        if not is_mosaic:
            print("Images don't match after ratio test because the percentage of good matches is less than 5%, abort")
            print()
            return image2
        else:
            return image2, image2
    else:
        if not is_mosaic:
            print("Images match, continue")
            print()

    # Step 4: Estimate fundamental matrix F using RANSAC
    F, F_mask, pts1, pts2 = fundamental_matrix(kp1, kp2, good_matches)

    # epipolar lines 
    if not is_mosaic:
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F).reshape(-1,3)
        img1,_ = drawlines(gray1,gray2,lines1,pts1,pts2)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F).reshape(-1,3)
        img2, _ = drawlines(gray2,gray1,lines2,pts2,pts1)

        output_path_img1 = os.path.join(directory, names[i] + "_with_epipolar_lines.jpg")
        output_path_img2 = os.path.join(directory, names[j] + "_with_epipolar_lines.jpg")
        cv2.imwrite(output_path_img1, img1)
        cv2.imwrite(output_path_img2, img2)



    # Step 5: Decision based on the number and percentage of inliers
    good_matches = np.array(good_matches)
    inliers = good_matches[F_mask.ravel()==1]
    if not is_mosaic:
        print("Number of inliers for fundamental_matrix:", len(inliers))
        print("Fraction of inliers:", len(inliers)/len(good_matches))
        print()

    if len(inliers)/len(good_matches) <= 0.05:
        if not is_mosaic:
            print("Images don't match after RANSAC because the percentage of inliners is less than 5%, abort")
            print()
            return image2
        else:
            return image2, image2
    else:
        if not is_mosaic:
            print("Images match, continue")
            print()

    # save keypoints
    if not is_mosaic:
        img_matches = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        input_files = [names[i], names[j]]
        output_filename = ("_".join(sorted(input_files))) + "_fundamental" + ".jpg"
        output_path = os.path.join(directory+"/", output_filename)
        cv2.imwrite(output_path, img_matches)


    # Part 6
    H, H_mask = cv2.findHomography(pts1, pts2, cv2.FM_RANSAC, confidence=0.99)
    H_inliers = good_matches[H_mask.ravel()==1]
    if not is_mosaic:
        print("Number of inliers for Homography matrix:", len(H_inliers))
        print("Fraction of inliers:", len(H_inliers)/len(good_matches))
        print()

    # Step 7: Decision based on the number and percentage of inliers for homography
    total_mask = F_mask[F_mask == H_mask]
    inliner_matches = total_mask[total_mask == 1]
    total_inliner_matches = len(inliner_matches)

    if total_inliner_matches/len(good_matches) <= 0.15:
        if not is_mosaic:
            print("Images don't match after findHomography because the percentage of inliners is less than 15%, abort")
            print()
            return image2
        else:
            return image2, image2
    else:
        if not is_mosaic:
            print("Images match, continue")
            print()

    # save keypoints
    if not is_mosaic:
        img_matches = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        input_files = [names[i], names[j]]
        output_filename = ("_".join(sorted(input_files))) + "_homography" + ".jpg"
        output_path = os.path.join(directory+"/", output_filename)
        cv2.imwrite(output_path, img_matches)


    #cropping
    mosaic1, mosaic2 = crop_img(image1, image2, H) 
    mosaic_img = mosaic2.copy()

    #blending
    mosaic_img[mosaic2 == 0] = mosaic1[mosaic2 == 0]
    mosaic_img_blended = Laplacian_pyramid_blend(mosaic1, mosaic2)
    # mosaic_img_blended = mosaic_img

    if is_mosaic:
        mosaic_img[mosaic2 == 0] = mosaic1[mosaic2 == 0]
        mosaic_img_blended = Laplacian_pyramid_blend(mosaic2, mosaic1)
        return mosaic_img, mosaic_img_blended
   

    # save image pairs
    if not is_mosaic:
        input_files = [names[i], names[j]]
        output_filename = ("_".join(sorted(input_files))) + "_pairs" + ".jpg"
        output_path = os.path.join(directory+"/", output_filename)
        cv2.imwrite(output_path, mosaic_img_blended)
    

    return mosaic_img_blended


if __name__ == "__main__":


    try:
        image_dir = sys.argv[1]
    except IndexError:
        # image_dir = 'hw3_data/drink-machine'
        # image_dir = 'hw3_data/frear-park'
        # image_dir = 'hw3_data/office'
        # image_dir = 'hw3_data/tree_mrc'
        # image_dir = 'hw3_data/vcc-entrance'
        image_dir = 'hw3_data/myplant'

        # python loshaa_Hw3.py hw3_data/drink-machine out_dir_drink-machine;
        # python loshaa_Hw3.py hw3_data/frear-park out_dir-frear-park;
        # python loshaa_Hw3.py hw3_data/office out_dir_office;
        # python loshaa_Hw3.py hw3_data/tree_mrc out_dir_tree_mrc;
        # python loshaa_Hw3.py hw3_data/vcc-entrance out_dir_vcc-entrance;
        # python loshaa_Hw3.py hw3_data/myplant out_dir_plant;

    try:
        directory = sys.argv[2]
    except IndexError:
        directory = "out_dir"

    if not os.path.exists(directory):
        os.makedirs(directory)

    print("image_dir:", image_dir)
    names, imgs = get_images(image_dir)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    print("num imgs:", len(imgs))
    print()

    mosaic_img = imgs[0]

    #Create img pairs
    for i, j in itertools.combinations(range(len(imgs)), 2):
        print(f"Img{i} and Img{j}")
        img_pairs = img_mosaic(i, j, imgs[j], imgs[i], names, imgs, sift, bf, directory)

    #Create mosaic    
    for i in range(1, len(imgs)):
        mosaic_img, mosaic_img_blended = img_mosaic(i, i, imgs[i], mosaic_img, names, imgs, sift, bf, directory, True)




    # Step 8: Build and output the mosaic of all images
    output_filename = "_".join(sorted(names)) + "_blended.jpg"
    cv2.imwrite(os.path.join(directory, output_filename), mosaic_img_blended)





    
