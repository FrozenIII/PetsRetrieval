function getImageInfo(url) {
  return new Promise((resolve, reject) => {
    wx.getImageInfo({
      src: url,
      success: resolve,
      fail: reject,
    })
  })
}

function createRpx2px() {
  const { windowWidth } = wx.getSystemInfoSync()

  return function(rpx) {
    return windowWidth / 750 * rpx
  }
}

const rpx2px = createRpx2px()

function canvasToTempFilePath(option, context) {
  return new Promise((resolve, reject) => {
    wx.canvasToTempFilePath({
      ...option,
      success: resolve,
      fail: reject,
    }, context)
  })
}

function saveImageToPhotosAlbum(option) {
  return new Promise((resolve, reject) => {
    wx.saveImageToPhotosAlbum({
      ...option,
      success: resolve,
      fail: reject,
    })
  })
}

Component({
  properties: {
    visible: {
      type: Boolean,
      value: false,
      observer(visible) {
        if (visible) {
          this.draw()
        }
      }
    },
    url: {
      type: String,
      value: ''
    },
    crop_width: Number,
    crop_height: Number,
    textArr: {
      type: Array,
      value: []
    },
    canvasWidth:  {
      type: Number,
      value: 843
    },
    canvasHeight: {
      type: Number,
      value: 1500
    },
  },

  data: {
    isDraw: false,

    canvasWidth: 843,
    canvasHeight: 1500,

    imageFile: '',

    responsiveScale: 1,
  },

  lifetimes: {
    ready() {
      const designWidth = 375
      const designHeight = 603 // 这是在顶部位置定义，底部无tabbar情况下的设计稿高度

      // 以iphone6为设计稿，计算相应的缩放比例
      const { windowWidth, windowHeight } = wx.getSystemInfoSync()
      const responsiveScale =
        windowHeight / ((windowWidth / designWidth) * designHeight)
      if (responsiveScale < 1) {
        this.setData({
          responsiveScale,
        })
      }
    },
  },

  methods: {
    handleClose() {
      this.triggerEvent('close')
    },
    handleSave() {
      const { imageFile } = this.data

      if (imageFile) {
        saveImageToPhotosAlbum({
          filePath: imageFile,
        }).then(() => {
          wx.showToast({
            icon: 'none',
            title: '分享图片已保存至相册',
            duration: 2000,
          })
        })
      }
    },
    draw() {
      wx.showLoading({
        title: '生成中...',
        mask: true,
      })
      this.setData({
        imageFile: ''
      })
      const { textArr, crop_height, crop_width, url, canvasWidth, canvasHeight } = this.data
      const avatarPromise = getImageInfo(url)
      const backgroundPromise = getImageInfo("https://dhhddd.mynatapp.cc/image_show?path=init_image/poster.jpg")
      Promise.all([avatarPromise, backgroundPromise])
        .then(([avatar, background]) => {
          const ctx = wx.createCanvasContext('share', this)

          const canvasW = rpx2px(canvasWidth * 2)
          const canvasH = rpx2px(canvasHeight * 2)
          console.log(avatar)
            // 绘制背景
          ctx.drawImage(
            background.path,
            0,
            0,
            canvasW,
            canvasH
          )

          // 绘制头像
          var width = rpx2px(crop_width * 2)
          var height = rpx2px(crop_height * 2)
          console.log(crop_width)
          console.log(crop_height)
          ctx.drawImage(
            avatar.path,
            (canvasW / 2) - (width / 2),
            canvasH / 2 - height,
            width,
            height
          )

          ctx.setTextAlign('center');
          ctx.setFillStyle('#342158');
          ctx.setFontSize(60);
          let contentHh = 50 * 1;
          for (let m = 0; m < textArr.length; m++) {
            if (height + rpx2px(200 * 2) + rpx2px(contentHh * m * 2) > canvasH) {
              break
            }
            ctx.fillText(textArr[m], canvasW / 2, height + rpx2px(200 * 2) + rpx2px(contentHh * m * 2));
          }

          ctx.stroke()

          ctx.draw(false, () => {
            canvasToTempFilePath({
              canvasId: 'share',
            }, this).then(({ tempFilePath }) => this.setData({ imageFile: tempFilePath }))
          })

          wx.hideLoading()
          this.setData({ isDraw: true })
        })
        .catch(() => {
          wx.hideLoading()
        })
    }
  }
})