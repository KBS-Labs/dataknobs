from dataknobs_data.records import Record

# Create a test record
r = Record(id="test1", data={"value": 100})
print("Record created:", r)
print("Record ID:", r.id)
print("Record to_dict():", r.to_dict())
print("Record to_dict(include_metadata=True):", r.to_dict(include_metadata=True))
