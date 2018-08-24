XmlPath = "C:/backend/binExe/configs/tlog_fields.xml"
# LogPath = "C:/backend/binExe/server/GameWorld/log/game.log"
LogPath = "F:/Online/shouxufei/game.log.4.match_word.20180816154922"
OutLogPath = "F:/Online/shouxufei/out.log"
# 是否使用过滤
UseFilter = False

OutToFile = True;
# 用不到的表和字段，请用'#'注释，不要删掉
# 需要查看的表和字段，在这里过滤,需要查看的字段请自行添加
TB_Filter = {
"Activity":["iActivityID","dtEventTime","vopenid","ChangedType","progress","param","param1"],
"TeamFlow":["vRoleID","dtEventTime","TeamOp"],
"TaskFlow":["vRoleID","dtEventTime","iTaskType","iTaskID","SubId","ChangedType","taskname","stepname"],

}