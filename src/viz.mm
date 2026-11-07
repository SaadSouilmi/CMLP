#import <AppKit/AppKit.h>
#include "viz.h"

static constexpr int scale = 10;

@interface GrayscaleView : NSView <NSWindowDelegate>
@property(nonatomic, assign) const uint8_t* pixels;
@property(nonatomic) int imageWidth;
@property(nonatomic) int imageHeight;
@end

@implementation GrayscaleView

- (BOOL)isFlipped {
    return YES;  // Put row 0 at the top.
}

- (void)drawRect:(NSRect)dirtyRect {
    for (int y = 0; y < self.imageHeight; ++y) {
        const uint8_t* row = self.pixels + (size_t)y * self.imageWidth;

        for (int x = 0; x < self.imageWidth; ++x) {
            CGFloat gray = row[x] / 255.0;
            [[NSColor colorWithDeviceWhite:gray alpha:1.0] setFill];
            NSRectFill(NSMakeRect((CGFloat)x * scale, (CGFloat)y * scale,
                                  scale, scale));
        }
    }
}

- (void)windowWillClose:(NSNotification*)notification {
    [NSApp stop:nil];
}

@end

void show_grayscale(const uint8_t* pixels, int width, int height) {
    if (!pixels || width <= 0 || height <= 0) return;

    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(0, 0, (CGFloat)width * scale,
                                        (CGFloat)height * scale);

        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                        backing:NSBackingStoreBuffered
                          defer:NO];

        window.releasedWhenClosed = NO;
        window.title = @"MNIST";

        GrayscaleView* view = [[GrayscaleView alloc] initWithFrame:frame];
        view.pixels = pixels;
        view.imageWidth = width;
        view.imageHeight = height;

        window.contentView = view;
        window.delegate = view;

        [window center];
        [window makeKeyAndOrderFront:nil];
        if (@available(macOS 14.0, *)) [app activate];
        [app run];

        window.delegate = nil;
        window.contentView = nil;
    }
}
