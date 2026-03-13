import MetaTrader5 as mt5

from config import DEVIATION, MAGIC
from data_mt5 import get_supported_filling


def place_market_order(symbol: str, side: int, volume: float, sl: float, tp: float):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return None

    filling_type = get_supported_filling(symbol)

    if side == 1:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": round(sl, info.digits),
        "tp": round(tp, info.digits),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "XAU_V7",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(request)
    if result is not None and result.retcode == 10030:
        for alt_fill in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
            if alt_fill == filling_type:
                continue
            request["type_filling"] = alt_fill
            result = mt5.order_send(request)
            if result is not None and result.retcode != 10030:
                break

    return result


def modify_position_sl_tp(symbol: str, ticket: int, sl: float = None, tp: float = None):
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None

    position = positions[0]
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": position.sl if sl is None else sl,
        "tp": position.tp if tp is None else tp,
    }
    return mt5.order_send(request)


def close_partial_position(symbol: str, position, close_volume: float):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": position.ticket,
        "volume": close_volume,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "XAU_V7_PARTIAL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_supported_filling(symbol),
    }
    return mt5.order_send(request)
