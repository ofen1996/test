from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext

# 创建一个数据块，用于存储寄存器的值
block = ModbusSequentialDataBlock(0, [0] * 100)

# 创建一个从站上下文，将数据块添加到从站上下文中
slave_context = ModbusSlaveContext(di=None, co=None, hr=block, ir=None)
context = ModbusServerContext(slaves=slave_context, single=True)

# 启动 Modbus TCP 服务端
StartTcpServer(context, address=('localhost', 502))