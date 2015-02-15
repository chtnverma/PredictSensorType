
import scipy, os, sys, pprint, json, 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import sklearn.svm as suppmach 


good_obj_datatypes = [0,1,2,3,4,5,13,14,19]
building = '517' 

def ReadInput(file): 
	global data 
	inp_file = open('discovered_devices.json')
	data=json.load(inp_file)

def GetFeatures(): 
	# Init: 
	names = [] 
	data_types = [] 
	object_desc_props= [] 
	jci_names= [] 
	units= [] 
	descs= [] 
	sensor_types= [] 
	props_type_strs= [] 
	props_types= [] 
	ids=[] 
	
	# Collect metadata from sensors: 
	for s_num in range(len(data[building]['objs'])): 
		if not int(data[building]['objs'][s_num]['data_type']) in good_obj_datatypes: continue 
		
		sensor_id= building + '_' + str(data[building]['objs'][s_num]['data_type']) + '_' + str(data[building]['objs'][s_num]['props']['instance'] 
		ids.append(sensor_id) 
		
		names.append(data[building]['objs'][s_num]['name']) 
		data_types.append({str(data[building]['objs'][s_num]['data_type']):1}) # categorical
		object_desc_props.append(data[building]['objs'][s_num]['object_desc_prop']) 
		#jci_names.append(data[building]['objs'][s_num]['jci_name'])  # ignoring since object_desc_props captures same keywords 
		units.append({str(data[building]['objs'][s_num]['unit']):1}) # categorical 
		#descs.append(data[building]['objs'][s_num]['desc']) # ignoring since sensor_type captures same tokens
		sensor_types.append(data[building]['objs'][s_num]['sensor_type'])
		props_type_strs.append(data[building]['objs'][s_num]['props']['type_str']) 
		props_types.append({str(data[building]['objs'][s_num]['props']['type']):1}) # categorical  
	
	print('Parsed metadata') 
	
	# Textual features to bag of words: 
	namevect = CountVectorizer() 
	namebow = scipy.sparse.coo_matrix(namevect.fit_transform(names))	
	object_desc_propsvect = CountVectorizer() 
	object_desc_propsbow = scipy.sparse.coo_matrix(object_desc_propsvect.fit_transform(object_desc_props))
	sensor_typesvect = CountVectorizer() 
	sensor_typesbow = scipy.sparse.coo_matrix(sensor_typesvect.fit_transform(sensor_types))
	props_type_strsvect = CountVectorizer() 
	props_type_strsbow = scipy.sparse.coo_matrix(props_type_strsvect.fit_transform(props_type_strs)) 
	
	# for categorical features: 
	data_typevect = DictVectorizer() 
	data_typebow = scipy.sparse.coo_matrix(data_typevect.transform(data_types)) 
	unitvect = DictVectorizer() 
	unitbow = scipy.sparse.coo_matrix(unitvect.transform(units)) 
	props_typevect = DictVectorizer() 
	props_typebow = scipy.sparse.coo_matrix(props_typevect.transform(props_types)) 
	print("obtained features") 
	
	# combine different feature vectors: 
	finalbow_matrix = scipy.sparse.hstack([namebow,object_desc_propsbow,sensor_typesbow,props_type_strsbow,data_typebow,unitbow,props_typebow]) 
	sensor_feature_map = { ids[i]:finalbow_matrix[i,:] for i in range(len(ids)) }   
	
	return sensor_feature_map 

def GetSensorsWithLabels(file): 
	f = open(file) 
	sensorid_groundtruth_map= {} 
	for line in f: 
		line_p = line.strip().split(',')  
		if len(line_p) == 2: 
			sensorid_groundtruth_map[line_p[0]] = line_p[1] 
	return(sensorid_groundtruth_map) 
		
		
def TrainEvaluateModel(sensorid_groundtruth_mapold,sensor_feature_mapold ):
	
	# remove sensors not having both features and labels. 
	common_sensors = set(sensorid_groundtruth_mapold.keys()) & set(sensor_feature_mapold.keys())
	sensor_feature_map = {i:sensor_feature_mapold[i] for i in common_sensors}
	sensorid_groundtruth_map = {i:sensorid_groundtruth_mapold[i] for i in common_sensors}
	print('Num of common sensors = %d' %len(common_sensors)) 
	
	featurebow = scipy.sparse.coo_matrix([])  
	labels = np.zeros(( len(common_sensors) ,)) 
	count=0 
	for s in sorted(common_sensors): 
		featurebow = scipy.sparse.vstack( [featurebow, sensor_feature_map[s]]) 
		labels[count] = sensorid_groundtruth_map[s] 
		count+=1
	print( "shape of labels = %s, of bow = %s" %(labels.shape , bow.shape))

def Syntax():
	print('Syntax: codename inp_jsonfile groundlabelcsv')
	
if __name__=='__main__': 
	if not len(sys.argv)==3: 
		Syntax()
		sys.exit(1) 		
	ReadInput(sys.argv[1]) 
	sensor_feature_map = GetFeatures() 
	sensorid_groundtruth_map = GetSensorsWithLabels(sys.argv[2]) 
	TrainEvaluateModel(sensorid_groundtruth_map,sensor_feature_map )

