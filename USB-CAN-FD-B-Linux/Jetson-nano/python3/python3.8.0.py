#
#python3.8.8 64位（python 32位要用32位的DLL）
#
# V2.0  -- add CAN frame send and recv. add close dev. add filter.
#
from ctypes import *
 
VCI_USBCAN2 = 41
STATUS_OK = 1
INVALID_DEVICE_HANDLE  = 0
INVALID_CHANNEL_HANDLE = 0
TYPE_CAN = 0
TYPE_CANFD = 1

class VCI_INIT_CONFIG(Structure):  
    _fields_ = [("AccCode", c_uint),
                ("AccMask", c_uint),
                ("Reserved", c_uint),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)
                ]  
class VCI_CAN_OBJ(Structure):  
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte*8),
                ("Reserved", c_ubyte*3)
                ] 
 
### structure
class _ZCAN_CHANNEL_CAN_INIT_CONFIG(Structure):
    _fields_ = [("acc_code", c_uint),
                ("acc_mask", c_uint),
                ("reserved", c_uint),
                ("filter",   c_ubyte),
                ("timing0",  c_ubyte),
                ("timing1",  c_ubyte),
                ("mode",     c_ubyte)]

class _ZCAN_CHANNEL_CANFD_INIT_CONFIG(Structure):
    _fields_ = [("acc_code",     c_uint),
                ("acc_mask",     c_uint),
                ("abit_timing",  c_uint),
                ("dbit_timing",  c_uint),
                ("brp",          c_uint),
                ("filter",       c_ubyte),
                ("mode",         c_ubyte),
                ("pad",          c_ushort),
                ("reserved",     c_uint)]

class _ZCAN_CHANNEL_INIT_CONFIG(Union):
    _fields_ = [("can", _ZCAN_CHANNEL_CAN_INIT_CONFIG), ("canfd", _ZCAN_CHANNEL_CANFD_INIT_CONFIG)]

class ZCAN_CHANNEL_INIT_CONFIG(Structure):
    _fields_ = [("can_type", c_uint),
                ("config", _ZCAN_CHANNEL_INIT_CONFIG)]
				
class ZCAN_CAN_FRAME(Structure):
    _fields_ = [("can_id",  c_uint, 29),
                ("err",     c_uint, 1),
                ("rtr",     c_uint, 1),
                ("eff",     c_uint, 1), 
                ("can_dlc", c_ubyte),
                ("__pad",   c_ubyte),
                ("__res0",  c_ubyte),
                ("__res1",  c_ubyte),
                ("data",    c_ubyte * 8)]

class ZCAN_CANFD_FRAME(Structure):
    _fields_ = [("can_id", c_uint, 29), 
                ("err",    c_uint, 1),
                ("rtr",    c_uint, 1),
                ("eff",    c_uint, 1), 
                ("len",    c_ubyte),
                ("brs",    c_ubyte, 1),
                ("esi",    c_ubyte, 1),
                ("__res",  c_ubyte, 6),
                ("__res0", c_ubyte),
                ("__res1", c_ubyte),
                ("data",   c_ubyte * 64)]

				
class ZCAN_Transmit_Data(Structure):
    _fields_ = [("frame", ZCAN_CAN_FRAME), ("transmit_type", c_uint)]

class ZCAN_Receive_Data(Structure):
    _fields_  = [("frame", ZCAN_CAN_FRAME), ("timestamp", c_ulonglong)]

class ZCAN_TransmitFD_Data(Structure):
    _fields_ = [("frame", ZCAN_CANFD_FRAME), ("transmit_type", c_uint)]

class ZCAN_ReceiveFD_Data(Structure):
    _fields_ = [("frame", ZCAN_CANFD_FRAME), ("timestamp", c_ulonglong)]

	
CanDLLName = './libcontrolcanfd.so' #把DLL放到对应的目录下
# canDLL = windll.LoadLibrary('./libcontrolcanfd.so')
#Linux系统下使用下面语句，编译命令：python3 cxcanfd.py
canDLL = cdll.LoadLibrary('/home/jetson/Music/python3.8.0/libcontrolcanfd.so')

print('########################################################')
print('## Chuang Xin USBCANFD python(x64) test program V2.0 ###')
print('########################################################')
print(CanDLLName)


canDLL.ZCAN_OpenDevice.restype = c_void_p
canDLL.ZCAN_SetAbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
canDLL.ZCAN_SetDbitBaud.argtypes = (c_void_p, c_ulong, c_ulong)
canDLL.ZCAN_SetCANFDStandard.argtypes = (c_void_p, c_ulong, c_ulong)
canDLL.ZCAN_InitCAN.argtypes = (c_void_p, c_ulong, c_void_p)
canDLL.ZCAN_InitCAN.restype = c_void_p
canDLL.ZCAN_StartCAN.argtypes = (c_void_p,)
canDLL.ZCAN_Transmit.argtypes = (c_void_p, c_void_p, c_ulong)
canDLL.ZCAN_TransmitFD.argtypes = (c_void_p, c_void_p, c_ulong)
canDLL.ZCAN_GetReceiveNum.argtypes = (c_void_p, c_ulong)
canDLL.ZCAN_Receive.argtypes = (c_void_p, c_void_p, c_ulong, c_long)
canDLL.ZCAN_ReceiveFD.argtypes = (c_void_p, c_void_p, c_ulong, c_long)
canDLL.ZCAN_ResetCAN.argtypes = (c_void_p,)
canDLL.ZCAN_CloseDevice.argtypes = (c_void_p,)

canDLL.ZCAN_ClearFilter.argtypes=(c_void_p,)
canDLL.ZCAN_AckFilter.argtypes=(c_void_p,)
canDLL.ZCAN_SetFilterMode.argtypes=(c_void_p,c_ulong)
canDLL.ZCAN_SetFilterStartID.argtypes=(c_void_p,c_ulong)
canDLL.ZCAN_SetFilterEndID.argtypes=(c_void_p,c_ulong)

#打开设备 
m_dev = canDLL.ZCAN_OpenDevice(VCI_USBCAN2, 0, 0)
if m_dev == INVALID_DEVICE_HANDLE:
    print("Open Device failed!")
    exit(0)
print("Open Device OK, device handle:0x%x." %(m_dev))
 

 
#设置通道0 仲裁域波特率：1M, 数据域波特率：5M
ret = canDLL.ZCAN_SetAbitBaud(m_dev,0,1000000)
if ret != STATUS_OK:
	print("Set CAN0 abit:1M failed!")
	exit(0)
print("Set CAN0 abit:1M OK!");
ret = canDLL.ZCAN_SetDbitBaud(m_dev,0,5000000)
if ret != STATUS_OK:
	print("Set CAN0 dbit:5M failed!")
	exit(0)
print("Set CAN0 dbit:5M OK!");

#设置通道1 仲裁域波特率：1M, 数据域波特率：5M
ret = canDLL.ZCAN_SetAbitBaud(m_dev,1,1000000)
if ret != STATUS_OK:
	print("Set CAN1 abit:1M failed!")
	exit(0)
print("Set CAN1 abit:1M OK!");
ret = canDLL.ZCAN_SetDbitBaud(m_dev,1,5000000)
if ret != STATUS_OK:
	print("Set CAN1 dbit:5M failed!")
	exit(0)
print("Set CAN1 dbit:5M OK!");

#设置通道0,1 FDMode : ISO
ret = canDLL.ZCAN_SetCANFDStandard(m_dev,0,0)
if ret != STATUS_OK:
	print("Set CAN0 ISO mode failed!")
	exit(0)
print("Set CAN0 ISO mode OK!")
ret = canDLL.ZCAN_SetCANFDStandard(m_dev,1,0)
if ret != STATUS_OK:
	print("Set CAN1 ISO mode failed!")
	exit(0)
print("Set CAN1 ISO mode OK!")



#初始0通道
init_config = ZCAN_CHANNEL_INIT_CONFIG()
init_config.can_type = TYPE_CANFD
init_config.config.canfd.mode = 0
dev_ch1 = canDLL.ZCAN_InitCAN(m_dev, 0, byref(init_config))
if dev_ch1 == INVALID_CHANNEL_HANDLE:
    print("Init CAN0 failed!")
    exit(0)
print("Init CAN0 OK!")

#启动通道0
ret = canDLL.ZCAN_StartCAN(dev_ch1)
if ret != STATUS_OK:
    print("Start CAN0 failed!")
    exit(0)
print("Start CAN0 OK!")	

#初始1通道
dev_ch2 = canDLL.ZCAN_InitCAN(m_dev, 1, byref(init_config))
if dev_ch2 == INVALID_CHANNEL_HANDLE:
    print("Init CAN1 failed!")
    exit(0)
print("Init CAN1 OK!")


###################################################################
#    在 ZCAN_InitCAN 之后， ZCAN_StartCAN之前配置
#	设置通道1  滤波:只接收 扩展帧,ID范围 5~6
###################################################################
canDLL.ZCAN_ClearFilter(dev_ch2)
canDLL.ZCAN_SetFilterMode(dev_ch2,1)
canDLL.ZCAN_SetFilterStartID(dev_ch2,5)
canDLL.ZCAN_SetFilterEndID(dev_ch2,6)
canDLL.ZCAN_AckFilter(dev_ch2)

#启动通道1
ret = canDLL.ZCAN_StartCAN(dev_ch2)
if ret != STATUS_OK:
    print("Start CAN1 failed!")
    exit(0)
print("Start CAN1 OK!")	


#################################
### CANFD frame send&&recv 
#################################
#通道1发送数据
transmit_canfd_num = 10
canfd_msgs = (ZCAN_TransmitFD_Data * transmit_canfd_num)()
for i in range(transmit_canfd_num):
	canfd_msgs[i].transmit_type = 0 #0=正常发送，1=单次发送，2=自发自收，3=单次自发自收。
	canfd_msgs[i].frame.eff     = 1 #extern frame
	canfd_msgs[i].frame.rtr     = 0 #remote frame
	canfd_msgs[i].frame.brs     = 1 #BRS 
	canfd_msgs[i].frame.can_id  = i
	canfd_msgs[i].frame.len     = 16
	for j in range(canfd_msgs[i].frame.len):
		canfd_msgs[i].frame.data[j] = j
ret = canDLL.ZCAN_TransmitFD(dev_ch1, canfd_msgs, transmit_canfd_num)
print("\r\n CAN0 Tranmit CANFD Num: %d." % ret)
 

#通道2接收数据
#Receive Messages
ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CANFD)
#print(ret)
while ret <= 0:#如果没有接收到数据，一直循环查询接收。
        ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CANFD)
if ret > 0:#接收到 ret 帧数据
	rcv_canfd_msgs = (ZCAN_ReceiveFD_Data * ret)()
	num = canDLL.ZCAN_ReceiveFD(dev_ch2, byref(rcv_canfd_msgs), ret, -1)
	print("CAN1 Received CANFD NUM: %d." % num)
	for i in range(num):
	    print("[%d]:ts:%d, id:%d, len:%d, eff:%d, rtr:%d, esi:%d, brs: %d, data:%s" %(
                        i, rcv_canfd_msgs[i].timestamp, rcv_canfd_msgs[i].frame.can_id, rcv_canfd_msgs[i].frame.len,
                        rcv_canfd_msgs[i].frame.eff, rcv_canfd_msgs[i].frame.rtr, 
                        rcv_canfd_msgs[i].frame.esi, rcv_canfd_msgs[i].frame.brs,
                        ''.join(str(rcv_canfd_msgs[i].frame.data[j]) + ' ' for j in range(rcv_canfd_msgs[i].frame.len))))

#################################
### CAN frame send&&recv
#################################
#通道1发送数据
transmit_can_num = 10
can_msgs = (ZCAN_Transmit_Data * transmit_can_num)()
for i in range(transmit_can_num):
	can_msgs[i].transmit_type = 0 #0=正常发送，1=单次发送，2=自发自收，3=单次自发自收。
	can_msgs[i].frame.eff     = 1 #extern frame
	can_msgs[i].frame.rtr     = 0 #remote frame
	#can_msgs[i].frame.brs     = 1 #BRS 
	can_msgs[i].frame.can_id  = i
	can_msgs[i].frame.can_dlc = 8
	for j in range(can_msgs[i].frame.can_dlc):
		can_msgs[i].frame.data[j] = j
ret = canDLL.ZCAN_Transmit(dev_ch1, can_msgs, transmit_can_num)
print("\r\n CAN0 Tranmit CAN Num: %d." % ret)
 

#通道2接收数据
#Receive Messages
ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CAN)
#print(ret)
while ret <= 0:#如果没有接收到数据，一直循环查询接收。
        ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CAN)
if ret > 0:#接收到 ret 帧数据
	rcv_can_msgs = (ZCAN_Receive_Data * ret)()
	num = canDLL.ZCAN_Receive(dev_ch2, byref(rcv_can_msgs), ret, -1)
	print("CAN1 Received CAN NUM: %d." % num)
	for i in range(num):
	    print("[%d]:ts:%d, id:%d, len:%d, eff:%d, rtr:%d, data:%s" %(
                        i, rcv_can_msgs[i].timestamp, rcv_can_msgs[i].frame.can_id, rcv_can_msgs[i].frame.can_dlc,
                        rcv_can_msgs[i].frame.eff, rcv_can_msgs[i].frame.rtr,                         
                        ''.join(str(rcv_can_msgs[i].frame.data[j]) + ' ' for j in range(rcv_can_msgs[i].frame.can_dlc))))


#关闭
ret = canDLL.ZCAN_ResetCAN(dev_ch1)
if ret != STATUS_OK:
    print("Close CAN0 failed!")
    exit(0)
print("Close CAN0 OK!")	

ret = canDLL.ZCAN_ResetCAN(dev_ch2)
if ret != STATUS_OK:
    print("Close CAN1 failed!")
    exit(0)
print("Close CAN1 OK!")	

ret = canDLL.ZCAN_CloseDevice(m_dev) 
if ret != STATUS_OK:
    print("Close Device failed!")
    exit(0)
print("Close Device OK!")
