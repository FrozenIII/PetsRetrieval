// pages/poster/poster.js
Page({
  data: {
    avater: "", //需要https图片路径
    //qrCode: "http://i4.hexun.com/2018-07-05/193365388.jpg", //需要https图片路径
    cname: '', //姓名
    description: "", //公司
    canvasWidth: 0,
    canvasHeight: 0
  },

  onLoad: function (options) {
    var imgUrl = wx.getStorageSync("imgUrl")
    var cname = wx.getStorageSync("cname")
    var description = wx.getStorageSync("description")
    var that = this;
    this.setData({
      avater: imgUrl,
      cname: cname,
      description: description
    })
    wx.getSystemInfo({
      
      //获取系统信息成功，将系统窗口的宽高赋给页面的宽高  
      success: function (res) {
        console.log(res)
        that.setData({
          canvasWidth: res.windowWidth - 50,
          canvasHeight: res.windowHeight - 100
        })
      }
    })
    
    that.sharePosteCanvas(imgUrl);
    //that.getAvaterInfo();
  },

  /**
   * 先下载头像图片
   */
  getAvaterInfo: function () {
    wx.showLoading({
      title: '生成中...',
      mask: true,
    });
    var that = this;
    wx.downloadFile({
      url: that.data.cardInfo.avater, //头像图片路径
      success: function (res) {
        wx.hideLoading();
        if (res.statusCode === 200) {
          var avaterSrc = res.tempFilePath; //下载成功返回结果
          //that.getQrCode(avaterSrc); //继续下载二维码图片
          that.sharePosteCanvas(avaterSrc);
        } else {
          wx.showToast({
            title: '头像下载失败！',
            icon: 'none',
            duration: 2000,
            success: function () {
              var avaterSrc = "";
              //that.getQrCode(avaterSrc);
              that.sharePosteCanvas(avaterSrc);
            }
          })
        }
      }
    })
  },

  /**
   * 下载二维码图片
   */
  getQrCode: function (avaterSrc) {
    wx.showLoading({
      title: '生成中...',
      mask: true,
    });
    var that = this;
    wx.downloadFile({
      url: that.data.cardInfo.qrCode, //二维码路径
      success: function (res) {
        wx.hideLoading();
        if (res.statusCode === 200) {
          var codeSrc = res.tempFilePath;
          that.sharePosteCanvas(avaterSrc, codeSrc);
        } else {
          wx.showToast({
            title: '二维码下载失败！',
            icon: 'none',
            duration: 2000,
            success: function () {
              var codeSrc = "";
              that.sharePosteCanvas(avaterSrc, codeSrc);
            }
          })
        }
      }
    })
  },

  /**
   * 开始用canvas绘制分享海报
   * @param avaterSrc 下载的头像图片路径
   * @param codeSrc   下载的二维码图片路径
   */
  sharePosteCanvas: function (avaterSrc, codeSrc) {
    wx.showLoading({
      title: '生成中...',
      mask: true,
    })
    var that = this;
    //var cardInfo = that.data.cardInfo; //需要绘制的数据集合
    var cname = that.data.cname;
    var description = that.data.description;
    const ctx = wx.createCanvasContext('myCanvas'); //创建画布
    var width = "";
    wx.createSelectorQuery().select('#canvas-container').boundingClientRect(function (rect) {
      var height = rect.height;
      var right = rect.right;
      width = rect.width * 0.5;
      var left = rect.left + 5;
      ctx.setFillStyle('#fff');
      ctx.fillRect(0, 0, rect.width, height);

      var clip_height = width;
      var clip_width = width;
      console.log(left)

      //头像为正方形
      if (avaterSrc) {
        wx.getImageInfo({
          src: avaterSrc,
          success(res) {
            console.log(res.width, res.height);
            console.log(width)
            var img_width = res.width,
                img_height = res.height;
            
            if (img_width > img_height) {
              clip_height = width * (img_height / img_width);
            }
            if (img_height > img_width) {
              clip_width = width * (img_width / img_height);
            }
            var locate = (rect.width - clip_width) / 2
            ctx.drawImage(avaterSrc, locate , 10, clip_width, clip_height);

            //姓名
      if (cname) {
        ctx.setFontSize(14);
        ctx.setFillStyle('#000');
        ctx.setTextAlign('center');
        var locate = (rect.width) / 2
        ctx.fillText(cname, locate, clip_height+10+rect.top);
      }

      if (description) {
        const CONTENT_ROW_LENGTH = 45; // 正文 单行显示字符长度
        let [contentLeng, contentArray, contentRows] = that.textByteLength(description, CONTENT_ROW_LENGTH);
        ctx.setTextAlign('center');
        ctx.setFillStyle('#000');
        ctx.setFontSize(12);
        let contentHh = 22 * 1;
        locate = (rect.width) / 2 
        for (let m = 0; m < contentArray.length; m++) {
          if (clip_height + 35 + rect.top + contentHh * m > rect.down) {
            break
          }
          ctx.fillText(contentArray[m],locate , clip_height + 35 + rect.top + contentHh * m);
        }
      }
          }
        })
        ;
      }

      

      //  绘制二维码
      if (codeSrc) {
        ctx.drawImage(codeSrc, left + 160, width + 40, width / 3, width / 3)
        ctx.setFontSize(10);
        ctx.setFillStyle('#000');
        ctx.fillText("微信扫码或长按识别", left + 160, width + 150);
      }

    }).exec()

    setTimeout(function () {
      ctx.draw();
      wx.hideLoading();
    }, 1000)

  },

  /**
   * 多行文字处理，每行显示数量
   * @param text 为传入的文本
   * @param num  为单行显示的字节长度
   */
  textByteLength(text, num) {
    let strLength = 0; // text byte length
    let rows = 1;
    let str = 0;
    let arr = [];
    for (let j = 0; j < text.length; j++) {
      if (text.charCodeAt(j) > 255) {
        strLength += 2;
        if (strLength > rows * num) {
          strLength++;
          arr.push(text.slice(str, j));
          str = j;
          rows++;
        }
      } else {
        strLength++;
        if (strLength > rows * num) {
          arr.push(text.slice(str, j));
          str = j;
          rows++;
        }
      }
    }
    arr.push(text.slice(str, text.length));
    return [strLength, arr, rows] //  [处理文字的总字节长度，每行显示内容的数组，行数]
  },

  //点击保存到相册
  saveShareImg: function () {
    var that = this;
    wx.showLoading({
      title: '正在保存',
      mask: true,
    })
    setTimeout(function () {
      wx.canvasToTempFilePath({
        canvasId: 'myCanvas',
        success: function (res) {
          wx.hideLoading();
          var tempFilePath = res.tempFilePath;
          wx.saveImageToPhotosAlbum({
            filePath: tempFilePath,
            success(res) {
              utils.aiCardActionRecord(19);
              wx.showModal({
                content: '图片已保存到相册，赶紧晒一下吧~',
                showCancel: false,
                confirmText: '好的',
                confirmColor: '#333',
                success: function (res) {
                  if (res.confirm) { }
                },
                fail: function (res) { }
              })
            },
            fail: function (res) {
              wx.showToast({
                title: res.errMsg,
                icon: 'none',
                duration: 2000
              })
            }
          })
        }
      });
    }, 1000);
  },

})