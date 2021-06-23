from tkinter import *
from flask import Flask,redirect, url_for,render_template,request
import os

def d_dtcn(num):
	
	os.system("python ml_model1.py "+num)
	
def ttm_func():
	os.system("python TTM1.py")

def ttm_exit():
	os.system("exit()")

'''root = Tk()
	root.configure(background = "white")

	def function1(): 
		os.system("python ml_model.py")
		exit()

	#def function2(): 
		#os.system("python android_cam.py --shape_predictor shape_predictor_68_face_landmarks.dat")
		#exit()

	
		
	root.title("DROWSINESS DETECTION")
	Label(root, text="Drowsiness Detection",font=("Verdana",20),fg="white",bg="black",height=2).grid(row=2,rowspan=2,columnspan=5,sticky=N+E+W+S,padx=5,pady=10)
	Button(root,text="Run using web cam",font=("Calibri",20),bg="#0D47A1",fg='white',command=function1).grid(row=5,columnspan=5,sticky=W+E+N+S,padx=50,pady=10)
	Button(root,text="Exit",font=("Calibri",20),bg="red",fg='white',command=root.destroy).grid(row=9,columnspan=5,sticky=W+E+N+S,padx=50,pady=10)

	root.mainloop()'''
