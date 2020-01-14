// pages/Searching/Searching.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    dataList: [], //数据源
    windowWidth: 0, //页面视图宽度
    windowHeight: 0, //视图高度
    imgMargin: 6, //图片边距: 单位px
    imgWidth: 0,  //图片宽度: 单位px
    topArr: [0, 0], //存储每列的累积top
    imgUrl: 'https://dhhddd.mynatapp.cc/image_show?path=init_image/deault.png',
    queryBase64: "",
    imgRes: [],
    desRes: [],
    caption: "",
    display: "block",
    listVis: false,
    crop_height: 0,
    crop_width: 0,
    visible: false,
    url: "",
    textArr: [],
    canvasWidth: 0,
    canvasHeight: 0,
    queryText: '',
    backTop: false,
    type: 1,
    searchType: 1
  },

  onReady: function () {
    
  },
  
  onShow: function(options) {
    var that = this;
    //获取页面宽高度
    wx.getSystemInfo({
      success: function (res) {
        console.log(res)

        var windowWidth = res.windowWidth;
        var imgMargin = that.data.imgMargin;
        //两列，每列的图片宽度
        var imgWidth = (windowWidth - imgMargin * 3) / 2;

        that.setData({
          windowWidth: windowWidth,
          windowHeight: res.windowHeight,
          imgWidth: imgWidth,
        });
        //!!!!!!!!!!
        //that.textByteLength("this pet is laughing hhhhhhhh", 40)
      },
    }),
      wx.getImageInfo({
        src: 'https://dhhddd.mynatapp.cc/image_show?path=init_image/poster.jpg',
        success: function (res) {
          that.setData({
            canvasWidth: res.width,
            canvasHeight: res.height
          })
        }
      })
  },

  upload: function(){
    var filePath = this.data.imgUrl;
    console.log(this.data.queryBase64)
    if (this.data.queryBase64.length == 0 && this.data.queryText.length == 0) {
      wx.showToast({
        title: '请重新选择一张照片或输入一个关键词',
        icon: 'none',
        duration: 2000
      })
    } else {
      wx.showLoading({
        title: '正在检索中...',
      });
      var that = this;
      console.log(this.data.queryText)
      if (this.data.queryBase64.length == 0) {
        this.setData({
          searchType: 2 //2表示仅用文字搜索
        })
      } else {
        this.setData({
          searchType: 1
        })
      }
      wx.request({
        url: "https://dhhddd.mynatapp.cc/retrieval_new", //仅为示例，并非真实的接口地址
        method: 'POST',
        data: {
          'queryImage': this.data.queryBase64,
          'num_result': 100,
          'queryText': this.data.queryText || "",
        },
        header: {
          'content-type': 'application/x-www-form-urlencoded'  // 默认值
        },
        success: function (response) {
          //输出的融合后的base64编码，不含前缀'data:image/jpeg;base64,'       
          if (response.statusCode === 200) {
            console.log(response.data)
            if (response.data.result_images.length == 0) {
              wx.hideLoading();
              wx.showToast({
                title: '请保证图片里面有萌宠噢',
                icon: 'none',
                duration: 2000
              })
              that.setData({
                value: '',
                queryBase64: '',
                queryText: ''
              })
            } else {
            that.setData({
              dataList: [],
              topArr: [0,0],
              type: 1,
              imgRes: response.data.result_images,
              caption: response.data.query_caption,
              desRes: response.data.result_captions,
              url: filePath,
              display: "None",
              listVis: true,
              value: '',
              queryBase64: '',
              queryText: ''
            });
            const CONTENT_ROW_LENGTH = 40;
            wx.hideLoading();
            that.textByteLength(response.data.query_caption, CONTENT_ROW_LENGTH)
            //console.log(imgRes);
            }
          }
          else {
            wx.hideLoading();
            wx.showToast({
              title: '与服务器连接失败，请重试',
              icon: 'none',
              duration: 2000
            })
          }
        },
        fail: function () {
          wx.hideLoading();
          wx.showToast({
            title: '与服务器连接失败',
            icon: 'none',
            duration: 2000
          })
        },
      })
    }
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function () {
    
  },

  reUpload: function () {
    var that = this;
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album', 'camera'],
      success: function (res) {
        console.log("wx.chooseImage album success")

        //const filePath = res.tempFilePaths[0]
        //console.log("filePath:" + filePath);
        //that.urlTobase64new(filePath)
        setTimeout(function() {
          wx.hideLoading()
        }, 200)
        that.setData({
          imgUrl: res.tempFilePaths[0]
        });
        that.urlTobase64new(res.tempFilePaths[0])
        //that.upload()
      },
      fail: e => {
        console.error(e);
      },
      complete: function (res) {
      },
    })
  },

  urlTobase64new(url) {
    var that = this
    wx.getFileSystemManager().readFile({
      filePath: url, //选择图片返回的相对路径
      encoding: 'base64', //编码格式
      success: res => {
        let base64 = res.data;
        that.setData({
          queryBase64: base64
        })
      },
      fail: function () {
        console.log(url)
        wx.showToast({
          title: '上传至服务器失败，请重新选择',
          icon: 'none',
          duration: 2000
        })
      },
    })
  },

  //加载图片
  loadImage: function (e) {
    console.log('execute loadimage')
    var index = e.currentTarget.dataset.index; //图片所在索引
    var imgW = e.detail.width, imgH = e.detail.height; //图片实际宽度和高度
    var imgWidth = this.data.imgWidth; //图片宽度
    var imgScaleH = imgWidth / imgW * imgH; //计算图片应该显示的高度

    var dataList = this.data.dataList;
    var margin = this.data.imgMargin;  //图片间距
    //第一列的累积top，和第二列的累积top
    var firtColH = this.data.topArr[0], secondColH = this.data.topArr[1];
    var obj = dataList[index];

    obj.height = imgScaleH;

    if (firtColH < secondColH) { //表示新图片应该放到第一列
      obj.left = margin;
      obj.top = firtColH + margin;
      firtColH += margin + obj.height + 20;
    }
    else { //放到第二列
      obj.left = margin * 2 + imgWidth;
      obj.top = secondColH + margin;
      secondColH += margin + obj.height + 20;
    }

    this.setData({
      dataList: dataList,
      topArr: [firtColH, secondColH],
    });
  },

  //加载更多图片
  loadMoreImages: function () {
    var imgs = this.data.imgRes;
    var captions = this.data.desRes;
    console.log('execute loadmoreimages')
    var tmpArr = [];
    console.log(imgs.length)
    for (let i = 0; i < Math.min(20, imgs.length); i++) {
      var index = i;
      var obj = {
        src: imgs[index],
        height: 0,
        top: 0,
        left: 0,
        caption: captions[index]
      }
      tmpArr.push(obj);
      imgs.splice(index, 1);
    }
    
    var dataList = this.data.dataList.concat(tmpArr)
    this.setData({ 
      dataList: dataList
      }, function () {
      wx.hideLoading()
    });
  },

  /**预览图片 */
  previewImg: function (e) {
    console.log(e)

    var index = e.currentTarget.dataset.index;
    var dataList = this.data.dataList;
    var currentSrc = dataList[index].src;
    var type = this.data.type;
    var that = this;
    if (type == 1) {
      wx.previewImage({
        urls: [currentSrc],
      })
    }
    if (type == 2) {
      that.setData({
        imgUrl: currentSrc,
        queryBase64: currentSrc
      })
    }
  },

  doShare: function() {
    var that = this;
    if (that.data.searchType == 2 || that.data.textArr.length == 0) {
      wx.showToast({
        title: '请先检索再分享照片哦～',
        icon: 'none',
        duration: 2000
      })
    } else {
      that.show()
    }
  },

  isChinese: function(s){
    var ret = false;
    for (var i = 0; i < s.length; i++) {
      ret = ret || (s.charCodeAt(i) >= 10000);
    }
    return ret;
  },

  //事件处理函数
  show: function () {
    var that = this;
    var imgPath = this.data.imgUrl;
    var canvasW = this.data.canvasWidth;
    var canvasH = this.data.canvasHeight;
    wx.getImageInfo({
      src: imgPath,
      success(res) {
        console.log(res)
        var img_width = res.width,
          img_height = res.height;
        var w_ratio = 1;
        var h_ratio = 1;
        if (img_width > img_height) {
          w_ratio = 0.8 * canvasW;
          h_ratio = (img_height / img_width) * w_ratio;
          if (h_ratio > canvasH * 0.4) {
            h_ratio = canvasH * 0.4;
            w_ratio = h_ratio * (img_width / img_height)
          }
        }
        if (img_height > img_width) {
          h_ratio = 0.4 * canvasH;
          w_ratio = (img_width / img_height) * h_ratio;
          if (w_ratio > canvasW * 0.8) {
            w_ratio = 0.8 * canvasW;
            h_ratio = w_ratio * (img_height / img_width)
          }
        }
        that.setData({
          crop_height: h_ratio,
          crop_width: w_ratio,
        })
        that.start()
      }
    })
  },

  textByteLength: function (text, num) {
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
    this.setData({
      textArr: arr
    })
    this.loadMoreImages()
  },

  start: function () {
    this.setData({ visible: true })
  },

  close: function () {
    this.setData({ visible: false })
  },


  doDownload: function() {
    var that = this;
    wx.request({
      url: "https://dhhddd.mynatapp.cc/get_querys", //仅为示例，并非真实的接口地址
      method: 'POST',
      data: {
        'num_result': 20,
      },
      header: {
        'content-type': 'application/x-www-form-urlencoded'  // 默认值
      },
      success: function (response) {
        //输出的融合后的base64编码，不含前缀'data:image/jpeg;base64,'       
        if (response.statusCode === 200) {
          console.log(response.data)
          that.setData({
            dataList: [],
            topArr: [0, 0],
            imgRes: response.data.result_images,
            desRes: response.data.result_captions,
            display: "None",
            textArr: [],
            listVis: true,
            type: 2
          });
          that.loadMoreImages()
        }
        else {
          wx.hideLoading();
          wx.showToast({
            title: '与服务器连接失败，请重试',
            icon: 'none',
            duration: 2000
          })
        }
      },
      fail: function () {
        wx.hideLoading();
        wx.showToasst({
          title: '与服务器连接失败',
          icon: 'none',
          duration: 2000
        })
      },
    })
  },

  onPageScroll: function (e) {
    var that = this
    var scrollTop = e.scrollTop
    var backTop = scrollTop > 100 ? true : false
    that.setData({
      backTop: backTop
    })
  },

  bindKeyInput: function(e){
    console.log(e)
    var value = e.detail.detail.value
    var res = this.isChinese(value)
    var that = this;
    if (res) {
      wx.showToast({
        title: '抱歉！当前仅支持英文关键词搜索',
        icon: 'none',
        duration: 2000
      })
      that.setData({
        queryText: ''
      })
    } else {
      that.setData({
        queryText: e.detail.detail.value
      })
    }
  }
})