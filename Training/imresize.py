import cv2

folder_path = '/home/captain_jack/Codes/OCR/Character_recog/'
print '============================================== FNT ================================================'
f = open('/home/captain_jack/Codes/OCR/k.out','w')
for i in range(62):
	for j in range(1016):
		#name = names[i].strip('\n')
		print '============= Sample : '+str(i) + ' image : '+str(j)+' =============='
		if j <= 715: 
			sub_folder = 'train/Fnt/'
			num = j+1
		elif j<= 865:
			sub_folder = 'valid/Fnt/'
			num = j - 715
		else:
			sub_folder = 'Test/Fnt/'
			num = j - 865
		char = '0'*(3-len(str(i+1)))+str(i+1)
		pho = '0'*(4-len(str(num)))+str(num)
		r = '0'*(5-len(str(j+1)))+str(j+1)
		name = 'Sample'+char+'/img'+char+'-'+pho
		im = cv2.imread(folder_path+'train/Fnt/Sample'+char+'/img'+char+'-'+r+'.png')
		#print folder_path+'train/Fnt/Sample'+char+'/img'+char+'-'+r+'.png'
		#print type(im)
		#name = name.split('-')[0] +'-'+'0'*(4-len(str(num)))+str(num)
		im1 = cv2.resize(im, (64,64))
		cv2.imwrite(folder_path+sub_folder+name+'_64'+'.png',im1)
		im1 = cv2.resize(im, (32,32))
		cv2.imwrite(folder_path+sub_folder+name+'_32'+'.png',im1)
		f.write(folder_path+sub_folder+name+'_64'+'.png\n')
		f.write(folder_path+sub_folder+name+'_32'+'.png\n')

f.close()
'''
for i in range(62):
	for j in range(55):
		#name = names[i].strip('\n')
		print '============= Sample : '+str(i) + ' image : '+str(j)+' =============='
		if j <= 39: 
			sub_folder = 'train/Hnd/'
			num = j+1
			print 'tr' +str(num)
		elif j<= 47:
			sub_folder = 'valid/Hnd/'
			num = j - 39
			print 'va' +str(num)
		else:
			sub_folder = 'Test/Hnd/'
			num = j - 47
			print 'te' +str(num)
		char = '0'*(3-len(str(i+1)))+str(i+1)
		pho = '0'*(3-len(str(num)))+str(num)
		r = '0'*(3-len(str(j+1)))+str(j+1)
		name = 'Sample'+char+'/img'+char+'-'+pho
		im = cv2.imread(folder_path+'train/Hnd/Img/Sample'+char+'/img'+char+'-'+r+'.png')
		#print folder_path+'train/Fnt/Sample'+char+'/img'+char+'-'+r+'.png'
		#print type(im)
		#name = name.split('-')[0] +'-'+'0'*(4-len(str(num)))+str(num)
		print folder_path+sub_folder+name+'_64'+'.png'
		im1 = cv2.resize(im, (64,64))
		cv2.imwrite(folder_path+sub_folder+name+'_64'+'.png',im1)
		im1 = cv2.resize(im, (32,32))
		cv2.imwrite(folder_path+sub_folder+name+'_32'+'.png',im1)

'''
