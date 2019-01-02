"""
Description: Module used to handle mongodb basic apis for mongodb basic operation
"""
__author__ = "pratik khandge"
__copyright__ = ""
__credits__ = ["pratik khandge"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "pratik"
__email__ = "pratik.khandge@gmail.com"
__status__ = "Developement"

# python imports
import pymongo
from gridfs import GridFS
from bson import ObjectId


class MongoConn(object):
    """
        Class to handle operations on Mongodb
    """

    def __init__(self, connection_obj, collection="test"):
        """
            Constructor of Mongo Connection.
        """
        self.database = connection_obj['DATABASE']
        self.collection = collection
        self.conn = pymongo.MongoClient(host=connection_obj['HOST'], port=connection_obj['PORT'])
        self.db_obj = self.conn[self.database]
        self.grid_fs = GridFS(self.db_obj)

    def __del__(self):
        if self.conn is not None:
            self.conn.close()

    def change_db_obj(self, database=None):
        """
            This method used to change database object
        """
        try:
            database = database if database else self.database
            self.db_obj = self.conn[database]
        except Exception as e:
            raise Exception("Could not change db object Exception: {}".format(e))

    def count(self, collection=None, filter=None):
        """
            This method used to get document counts
        """
        try:
            collection = collection if collection else self.collection
            filter = filter if isinstance(filter, dict) else {}
            result = self.db_obj[collection].find(filter).count()
        except Exception as e:
            result = 0
            raise Exception("Could not count documents in collection {} Exception: {}".format(collection, e))
        finally:
            return result

    def is_duplicate(self, doc, collection=None):
        """
            This method checks if object to be created already exists.
        """
        try:
            collection = collection if collection else self.collection
            return self.db_obj[collection].find_one(doc)
        except Exception as e:
            raise Exception("Error while checking duplicate documents in collection {} Exception: {}".format(collection, e))

    def list_document(self, collection=None, filter=None, sort=None, include=None, exclude=None, skip=0, limit=0):
        """
            This method retrieves list of documents with including only necessary
            fields provided in Include array.
            If Include array is empty then method retrieves document excluding
            the fields provided in Exclude array.
            If both Include and Exclude arrays are empty then method retrieves
            whole document.
        """
        try:
            filter = filter if isinstance(filter, dict) else {}
            sort = sort if sort else [("_id", 1)]
            include = include if include else []
            exclude = exclude if exclude else []
            collection = collection if collection else self.collection
            skip = int(skip) if type(skip) == int else 0
            limit = int(limit) if type(limit) == int else 0
            if len(include):
                select = dict([(key, 1) for key in include])
            elif len(exclude):
                select = dict([(key, 0) for key in exclude])
            else:
                select = None
            results = [result for result in
                       self.db_obj[collection].find(filter, select).sort(sort).skip(skip).limit(limit)]
            for ind, val in enumerate(results):
                results[ind]["id"] = str(results[ind].pop("_id"))
        except Exception as e:
            results = []
            raise Exception("Could not find documents in collection {} Exception: {}".format(collection, e))
        finally:
            return results
    def write_document(self, collection=None, filter=None, sort=None, include=None, exclude=None, skip=0, limit=0):
        """
            This method retrieves list of documents with including only necessary
            fields provided in Include array.
            If Include array is empty then method retrieves document excluding
            the fields provided in Exclude array.
            If both Include and Exclude arrays are empty then method retrieves
            whole document.
        """
        try:
            filter = filter if isinstance(filter, dict) else {}
            sort = sort if sort else [("_id", 1)]
            include = include if include else []
            exclude = exclude if exclude else []
            collection = collection if collection else self.collection
            skip = int(skip) if type(skip) == int else 0
            limit = int(limit) if type(limit) == int else 0
            if len(include):
                select = dict([(key, 1) for key in include])
            elif len(exclude):
                select = dict([(key, 0) for key in exclude])
            else:
                select = None
            results = [result for result in
                       self.db_obj[collection].find(filter, select).batch_size(500).sort(sort).skip(skip).limit(limit)]
            for ind, val in enumerate(results):
                results[ind]["id"] = str(results[ind].pop("_id"))
        except Exception as e:
            results = []
            raise Exception("Could not find documents in collection {} Exception: {}".format(collection, e))
        finally:
            return results

    def read_document(self, doc_id=None, filter=None, collection=None, include=None, exclude=None):
        """
            This method retrieves only one document with including only necessary
            fields provided in Include array.
            If Include array is empty then method retrieves document excluding
            the fields provided in Exclude array.
            If both Include and Exclude arrays are empty then method retrieves
            whole document.
        """
        try:
            if isinstance(filter, dict) and doc_id:
                filter["_id"] = ObjectId(doc_id)
            elif isinstance(filter, dict):
                filter = filter
            else:
                filter = {"_id": ObjectId(doc_id)}
            include = include if include else []
            exclude = exclude if exclude else []
            collection = collection if collection else self.collection
            if len(include):
                select = dict([(key, 1) for key in include])
            elif len(exclude):
                select = dict([(key, 0) for key in exclude])
            else:
                select = None
            result = self.db_obj[collection].find_one(filter, select)
        except Exception as e:
            result = {}
            raise Exception("Could not find documents in collection {} Exception: {}".format(collection, e))
        finally:
            return result

    def insert_document(self, doc, collection=None):
        """
            This method inserts given document in the provided collection of
            database object.
        """
        try:
            collection = collection if collection else self.collection
            return self.db_obj[collection].insert(doc)
        except Exception as  e:
            raise Exception("Could not insert document in collection {} Exception: {}".format(collection, e))

    def update_document(self, doc, doc_id, collection=None):
        """
            This method updates given document in the provided collection of
            database object.
        """
        try:
            collection = collection if collection else self.collection
            return self.db_obj[collection].update({"_id": ObjectId(doc_id)}, {"$set": doc})
        except Exception as e:
            raise Exception("Could not update document in collection {} Exception: {}".format(collection, e))

    def delete_document(self, doc_id=None, collection=None, filter=None):
        """
            This method deletes document by id in the provided collection of
            database object.
        """
        try:
            collection = collection if collection else self.collection
            filter = filter if isinstance(filter, dict) else {"_id": ObjectId(doc_id)}
            return self.db_obj[collection].remove(filter)
        except Exception as e:
            raise Exception("Could not delete document from collection {} Exception: {}".format(collection, e))

    def distinct(self, field, collection=None, filter=None):
        """
            Returns distinct values for that perticular field;
        """
        try:
            filter = filter if isinstance(filter, dict) else {}
            collection = collection if collection else self.collection
            if filter:
                return self.db_obj[collection].find(filter).distinct(field)
            else:
                return self.db_obj[collection].distinct(field)
        except Exception as e:
            raise Exception("Could not find distinct document in collection {} Exception: {}".format(collection, e))

    def upload_file(self, body, content_type, filename, Name):
        """
           Uploades file into gridfs.
        """
        try:
            return self.grid_fs.put(body, content_type=content_type, filename=filename, Name=Name)
        except Exception as e:
            raise Exception("Could not update file Exception: {}".format(e))

    def read_file(self, doc_id):
        """
           Reads file from gridfs.
        """
        try:
            return self.grid_fs.get(ObjectId(doc_id))
        except Exception as e:
            raise Exception("Could read file Exception: {}".format(e))

    def delete_file(self, doc_id):
        """
           Deletes file from gridfs.
        """
        try:
            return self.grid_fs.delete(ObjectId(doc_id))
        except Exception as e:
            raise Exception("Could delete file Exception: {}".format(e))

    def insert_document_unique_id(self, doc, inc_collection, unique_field, collection):
        """
        This function save data with unique field
        """
        doc[unique_field] = self.db_obj[inc_collection].find_and_modify(
            query={'colection': 'admin_colection'},
            update={'$inc': {'id': 1}},
            fields={'id': 1, '_id': 0},
            new=True
        ).get('id')

        try:
            self.db_obj[collection].insert(doc)
            return doc
        except pymongo.errors.DuplicateKeyError as e:
            self.insert_document_unique_id(doc)
