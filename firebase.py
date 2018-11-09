import pyrebase as pb
import firebase_admin
from firebase_admin import credentials

config = {
    "apiKey": "AIzaSyD-UebDzKViJhA9cVLSNRNK3JTLxTvSkLY",
    "authDomain": "ttu-usli-drone.firebaseapp.com",
    "databaseURL": "https://ttu-usli-drone.firebaseio.com",
    "storageBucket": "ttu-usli-drone.appspot.com",
    "serviceAccount": "ttu-usli-drone-firebase-adminsdk-3ve40-54184f8f79.json"
  }



class dataBase():
    def init(self):
        self.firebase = pb.initialize_app(config)
        self.db = self.firebase.database()

    def storeColors(self,min,sel,max):

        data = {
            "colors":{"min":{"H":int(min[0]),"S":int(min[1]),"V":int(min[2])},
                      "max":{"H":int(max[0]),"S":int(max[1]),"V":int(max[2])},
                      "sel":{"H":int(sel[0]),"S":int(sel[1]),"V":int(sel[2])}}
        }
        results = self.db.child("flight").update(data)
        #print(results)

    def storeCentroid(self,centroid):
        data = {
            "centroid":{
            "x":centroid[0],
            "y":centroid[1]
            }
        }
        results = self.db.child("flight").update(data)
        #print(results)