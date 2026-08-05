from ctypes import *
import time

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

CanDLLName = './libcontrolcanfd.so'
canDLL = cdll.LoadLibrary(CanDLLName)

# Define function argument and return types
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

canDLL.ZCAN_ClearFilter.argtypes = (c_void_p,)
canDLL.ZCAN_AckFilter.argtypes = (c_void_p,)
canDLL.ZCAN_SetFilterMode.argtypes = (c_void_p, c_ulong)
canDLL.ZCAN_SetFilterStartID.argtypes = (c_void_p, c_ulong)
canDLL.ZCAN_SetFilterEndID.argtypes = (c_void_p, c_ulong)

def open_device():
    m_dev = canDLL.ZCAN_OpenDevice(VCI_USBCAN2, 0, 0)
    if m_dev == INVALID_DEVICE_HANDLE:
        print("Open Device failed!")
        exit(0)
    print("Open Device OK, device handle:0x%x." % m_dev)
    return m_dev

def set_baud_rate(device_handle):
    # Set baud rate for CAN0 and CAN1
    for channel in range(2):
        ret = canDLL.ZCAN_SetAbitBaud(device_handle, channel, 1000000)
        if ret != STATUS_OK:
            print(f"Set CAN{channel} abit:1M failed!")
            exit(0)
        print(f"Set CAN{channel} abit:1M OK!")
        ret = canDLL.ZCAN_SetDbitBaud(device_handle, channel, 5000000)
        if ret != STATUS_OK:
            print(f"Set CAN{channel} dbit:5M failed!")
            exit(0)
        print(f"Set CAN{channel} dbit:5M OK!")

def configure_canfd_mode(device_handle):
    for channel in range(2):
        ret = canDLL.ZCAN_SetCANFDStandard(device_handle, channel, 0)
        if ret != STATUS_OK:
            print(f"Set CAN{channel} ISO mode failed!")
            exit(0)
        print(f"Set CAN{channel} ISO mode OK!")

def init_channel(device_handle, channel):
    init_config = ZCAN_CHANNEL_INIT_CONFIG()
    init_config.can_type = TYPE_CANFD
    init_config.config.canfd.mode = 0
    dev_ch = canDLL.ZCAN_InitCAN(device_handle, channel, byref(init_config))
    if dev_ch == INVALID_CHANNEL_HANDLE:
        print(f"Init CAN{channel} failed!")
        exit(0)
    print(f"Init CAN{channel} OK!")
    return dev_ch

def start_channel(dev_ch):
    ret = canDLL.ZCAN_StartCAN(dev_ch)
    if ret != STATUS_OK:
        print(f"Start CAN channel failed!")
        exit(0)
    print("Start CAN channel OK!")

def configure_filter(dev_ch2):
    canDLL.ZCAN_ClearFilter(dev_ch2)
    canDLL.ZCAN_SetFilterMode(dev_ch2, 1)
    canDLL.ZCAN_SetFilterStartID(dev_ch2, 5)
    canDLL.ZCAN_SetFilterEndID(dev_ch2, 6)
    canDLL.ZCAN_AckFilter(dev_ch2)

def send_canfd_data(dev_ch1):
    transmit_canfd_num = 10
    canfd_msgs = (ZCAN_TransmitFD_Data * transmit_canfd_num)()
    for i in range(transmit_canfd_num):
        canfd_msgs[i].transmit_type = 0
        canfd_msgs[i].frame.eff     = 1
        canfd_msgs[i].frame.rtr     = 0
        canfd_msgs[i].frame.brs     = 1
        canfd_msgs[i].frame.can_id  = i
        canfd_msgs[i].frame.len     = 16
        for j in range(canfd_msgs[i].frame.len):
            canfd_msgs[i].frame.data[j] = j
    ret = canDLL.ZCAN_TransmitFD(dev_ch1, canfd_msgs, transmit_canfd_num)
    print(f"\nCAN0 Transmit CANFD Num: {ret}.")

def receive_canfd_data(dev_ch2):
    ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CANFD)
    while ret <= 0:
        time.sleep(0.01)  # Add a small delay to avoid busy-waiting
        ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CANFD)
    if ret > 0:
        rcv_canfd_msgs = (ZCAN_ReceiveFD_Data * ret)()
        num = canDLL.ZCAN_ReceiveFD(dev_ch2, byref(rcv_canfd_msgs), ret, -1)
        print(f"CAN1 Received CANFD NUM: {num}.")
        for i in range(num):
            print(f"[{i}]:ts:{rcv_canfd_msgs[i].timestamp}, id:{rcv_canfd_msgs[i].frame.can_id}, len:{rcv_canfd_msgs[i].frame.len}, "
                  f"eff:{rcv_canfd_msgs[i].frame.eff}, rtr:{rcv_canfd_msgs[i].frame.rtr}, esi:{rcv_canfd_msgs[i].frame.esi}, "
                  f"brs:{rcv_canfd_msgs[i].frame.brs}, data:{' '.join(str(rcv_canfd_msgs[i].frame.data[j]) for j in range(rcv_canfd_msgs[i].frame.len))}")

def send_can_data(dev_ch1):
    transmit_can_num = 10
    can_msgs = (ZCAN_Transmit_Data * transmit_can_num)()
    for i in range(transmit_can_num):
        can_msgs[i].transmit_type = 0
        can_msgs[i].frame.eff     = 1
        can_msgs[i].frame.rtr     = 0
        can_msgs[i].frame.can_id  = i
        can_msgs[i].frame.can_dlc = 8
        for j in range(can_msgs[i].frame.can_dlc):
            can_msgs[i].frame.data[j] = j
    ret = canDLL.ZCAN_Transmit(dev_ch1, can_msgs, transmit_can_num)
    print(f"\nCAN0 Transmit CAN Num: {ret}.")

def receive_can_data(dev_ch2):
    ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CAN)
    while ret <= 0:
        time.sleep(0.01)  # Add a small delay to avoid busy-waiting
        ret = canDLL.ZCAN_GetReceiveNum(dev_ch2, TYPE_CAN)
    if ret > 0:
        rcv_can_msgs = (ZCAN_Receive_Data * ret)()
        num = canDLL.ZCAN_Receive(dev_ch2, byref(rcv_can_msgs), ret, -1)
        print(f"CAN1 Received CAN NUM: {num}.")
        for i in range(num):
            print(f"[{i}]:ts:{rcv_can_msgs[i].timestamp}, id:{rcv_can_msgs[i].frame.can_id}, len:{rcv_can_msgs[i].frame.can_dlc}, "
                  f"eff:{rcv_can_msgs[i].frame.eff}, rtr:{rcv_can_msgs[i].frame.rtr}, "
                  f"data:{' '.join(str(rcv_can_msgs[i].frame.data[j]) for j in range(rcv_can_msgs[i].frame.can_dlc))}")

def close_device(dev_ch1, dev_ch2, device_handle):
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
    ret = canDLL.ZCAN_CloseDevice(device_handle)
    if ret != STATUS_OK:
        print("Close Device failed!")
        exit(0)
    print("Close Device OK!")

def main():
    print('########################################################')
    print('## Chuang Xin USBCANFD python(x64) test program V2.0 ###')
    print('########################################################')
    print(CanDLLName)

    device_handle = open_device()
    set_baud_rate(device_handle)
    configure_canfd_mode(device_handle)
    
    dev_ch1 = init_channel(device_handle, 0)
    start_channel(dev_ch1)
    
    dev_ch2 = init_channel(device_handle, 1)
    configure_filter(dev_ch2)
    start_channel(dev_ch2)

    send_canfd_data(dev_ch1)
    receive_canfd_data(dev_ch2)

    send_can_data(dev_ch1)
    receive_can_data(dev_ch2)

    close_device(dev_ch1, dev_ch2, device_handle)

if __name__ == "__main__":
    main()
