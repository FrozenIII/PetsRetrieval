//index.js

//获取应用实例
const app = getApp()

Page({
  data: {

  },
  
  doSearchImage: function() {
    wx.navigateTo({
      url: '../Searching/Searching',
      //url: 'Search?filePath=' + filePath
    });
  }
})
