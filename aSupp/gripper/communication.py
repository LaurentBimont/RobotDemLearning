
class Communication:
   def __init__(self):
      self.client = None
      self.lock = threading.Lock()

   def connectToDevice(self, address):
      """Connection to the client - the method takes the IP address (as a string, e.g. '192.168.1.11') as an argument."""
      self.client = ModbusTcpClient(address)

   def disconnectFromDevice(self):
      """Close connection"""
      self.client.close()

   def sendCommand(self, data):
      """Send a command to the Gripper - the method takes a list of uint8 as an argument. The meaning of each variable depends on the Gripper model (see support.robotiq.com for more details)"""
      #make sure data has an even number of elements
      if(len(data) % 2 == 1):
         data.append(0)

      #Initiate message as an empty list
      message = []

      #Fill message by combining two bytes in one register
      for i in range(0, len(data)/2):
         message.append((data[2*i] << 8) + data[2*i+1])

      #To do!: Implement try/except
      with self.lock:
         self.client.write_registers(0, message)

   def getStatus(self, numBytes):
      """Sends a request to read, wait for the response and returns the Gripper status. The method gets the number of bytes to read as an argument"""
      numRegs = int(ceil(numBytes/2.0))

      #To do!: Implement try/except
      #Get status from the device
      with self.lock:
         response = self.client.read_input_registers(0, numRegs)

      #Instantiate output as an empty list
      output = []

      #Fill the output with the bytes in the appropriate order
      for i in range(0, numRegs):
         output.append((response.getRegister(i) & 0xFF00) >> 8)
         output.append( response.getRegister(i) & 0x00FF)

      #Output the result
      return output
