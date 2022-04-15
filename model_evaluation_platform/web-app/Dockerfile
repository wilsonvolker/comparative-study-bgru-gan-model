FROM node:17.7.1-alpine3.15

RUN apk add --no-cache libc6-compat

# RUN npm i -g yarn

ENV NODE_ENV production
ENV PORT 3000

EXPOSE 3000

WORKDIR /home/nextjs/app

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

COPY package.json .
#COPY package-lock.json .

RUN chown -R nextjs:nodejs /home/nextjs

USER nextjs

RUN yarn install
RUN npx browserslist@latest --update-db
RUN npx next telemetry disable

COPY . .

ENV NEXT_PUBLIC_EVALUATE_URL=.../evaluate
ENV NEXT_PUBLIC_DEFAULT_STOCKS_URL=.../default_stocks

RUN yarn run build

CMD [ "yarn", "start" ]