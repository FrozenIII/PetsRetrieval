//域名地址
var prefix = 'http://dhhddd.mynatapp.cc';
//接口授权码
var authCode = '自己的接口授权码';
//接口访问类型
var clientType = 'wsc';

const visionimgfilterurl = prefix + '/rest/ptu/visionimgfilter?clientType=' + clientType + '&authCode=' + authCode;

function getVsionimgfilterurl() {
  return visionimgfilterurl;
}

module.exports.getVsionimgfilterurl = getVsionimgfilterurl;