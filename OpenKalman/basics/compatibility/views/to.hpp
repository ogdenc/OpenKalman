/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref ranges::to.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_TO_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_TO_HPP

#include "basics/compatibility/language-features.hpp"
#include "view-concepts.hpp"
#include "range_adaptor_closure.hpp"
#include "transform.hpp"

namespace OpenKalman::stdex::ranges
{
#if __cpp_lib_ranges_to_container >= 202202L
  using std::ranges::to;
#else
  namespace detail_to
  {
    using namespace std;


#ifdef __cpp_concepts
    template<typename Adaptor, typename...Args>
    concept adaptor_invocable = requires { std::declval<Adaptor>()(declval<Args>()...); };
#else
    template<typename Adaptor, typename = void, typename...Args>
    struct is_adaptor_invocable : std::false_type {};

    template<typename Adaptor, typename...Args>
    struct is_adaptor_invocable<Adaptor, std::void_t<decltype(std::declval<Adaptor>()(declval<Args>()...))>, Args...>
      : std::true_type {};

    template<typename Adaptor, typename...Args>
    inline constexpr bool adaptor_invocable = is_adaptor_invocable<Adaptor, void, Args...>::value;
#endif

    
    template<typename Adaptor, typename...Args>
    struct Partial;

    
    template<typename Adaptor, typename...Args>
    struct Partial : stdex::ranges::range_adaptor_closure<Partial<Adaptor, Args...>>
    {
      template<typename...Ts>
	    constexpr explicit
  	  Partial(int, Ts&&...args) : m_args(std::forward<Ts>(args)...) {}


#ifdef __cpp_concepts
      template<typename Range> requires adaptor_invocable<Adaptor, Range&&, const Args&...>
#else
      template<typename Range, std::enable_if_t<adaptor_invocable<Adaptor, Range&&, const Args&...>, int> = 0>
#endif
	    constexpr auto
	    operator()(Range&& r) const &
	    {
	      auto forwarder = [&r] (const auto&...args) { return Adaptor{}(std::forward<Range>(r), args...); };
	      return std::apply(forwarder, m_args);
	    }


#ifdef __cpp_concepts
      template<typename Range> requires adaptor_invocable<Adaptor, Range, Args...>
#else
      template<typename Range, std::enable_if_t<adaptor_invocable<Adaptor, Range, Args...>, int> = 0>
#endif
      constexpr auto
	    operator()(Range&& r) &&
	    {
	      auto forwarder = [&r] (auto&... args) { return Adaptor{}(std::forward<Range>(r), std::move(args)...); };
	      return std::apply(forwarder, m_args);
	    }


      template<typename Range>
	    constexpr auto
	    operator()(Range&& r) const && = delete;

    private:
      
      tuple<Args...> m_args;

    };

    
    template<typename Adaptor, typename Arg>
    struct Partial<Adaptor, Arg> : stdex::ranges::range_adaptor_closure<Partial<Adaptor, Arg>>
    {
      template<typename Tp>
	    constexpr
	    Partial(int, Tp&& arg) : m_arg(std::forward<Tp>(arg)) {}


#ifdef __cpp_concepts
      template<typename Range> requires adaptor_invocable<Adaptor, Range, const Arg&>
#else
      template<typename Range, std::enable_if_t<adaptor_invocable<Adaptor, Range, const Arg&>, int> = 0>
#endif
	    constexpr auto
	    operator()(Range&& r) const & { return Adaptor{}(std::forward<Range>(r), m_arg); }


#ifdef __cpp_concepts
      template<typename Range> requires adaptor_invocable<Adaptor, Range, Arg>
#else
      template<typename Range, std::enable_if_t<adaptor_invocable<Adaptor, Range, Arg>, int> = 0>
#endif
      constexpr auto
	    operator()(Range&& r) && { return Adaptor{}(std::forward<Range>(r), std::move(m_arg)); }


      template<typename Range>
	    constexpr auto
	    operator()(Range&& r) const && = delete;
      
    private:
      
      Arg m_arg;
    };


#ifdef __cpp_lib_ranges
    template<typename Container>
    inline constexpr bool reservable_container = std::ranges::sized_range<Container> and
      requires(Container& c, std::ranges::range_size_t<Container> n)
    {
      c.reserve(n);
      { c.capacity() } -> std::same_as<decltype(n)>;
      { c.max_size() } -> std::same_as<decltype(n)>;
    };

    template<typename Cont, typename Range>
    inline constexpr bool toable = requires {
      requires (not std::ranges::input_range<Cont> or
        std::convertible_to<std::ranges::range_reference_t<Range>, std::ranges::range_value_t<Cont>>);
    };
#else
    template<typename T, typename = void, typename = void>
    struct reservable_container_impl : std::false_type {};

    template<typename T>
    struct reservable_container_impl<T, std::void_t<decltype(std::declval<T&>().reserve(std::declval<stdex::ranges::range_size_t<T>>()))>,
      std::enable_if_t<
        std::is_same_v<decltype(std::declval<T&>().capacity()), stdex::ranges::range_size_t<T>> and
        std::is_same_v<decltype(std::declval<T&>().max_size()), stdex::ranges::range_size_t<T>>>>
      : std::true_type {};

    template<typename Container>
    inline constexpr bool reservable_container =
      stdex::ranges::sized_range<Container> and reservable_container_impl<Container>::value;

    template<typename Range, typename = void>
    struct toable1_impl : std::false_type {};

    template<typename Cont>
    struct toable1_impl<Cont, std::enable_if_t<not stdex::ranges::input_range<Cont>>> : std::true_type {};

    template<typename Cont, typename Range, typename = void>
    struct toable2_impl : std::false_type {};

    template<typename Cont, typename Range>
    struct toable2_impl<Cont, Range, std::enable_if_t<
      stdex::convertible_to<stdex::ranges::range_reference_t<Range>, stdex::ranges::range_value_t<Cont>>>> : std::true_type {};

    template<typename Cont, typename Range>
    inline constexpr bool toable = toable1_impl<Cont>::value or toable2_impl<Cont, Range>::value;
#endif
#ifndef __cpp_concepts
    template<typename C, typename I, typename = void>
    struct can_emplace_back : std::false_type {};

    template<typename C, typename I>
    struct can_emplace_back<C, I, std::void_t<decltype(std::declval<C&>().emplace_back(*std::declval<I&>()))>>
      : std::true_type {};


    template<typename C, typename I, typename = void>
    struct can_push_back : std::false_type {};

    template<typename C, typename I>
    struct can_push_back<C, I, std::void_t<decltype(std::declval<C&>().push_back(*std::declval<I&>()))>>
      : std::true_type {};


    template<typename C, typename I, typename = void>
    struct can_emplace : std::false_type {};

    template<typename C, typename I>
    struct can_emplace<C, I, std::void_t<decltype(std::declval<C&>().emplace(std::declval<C&>().end(), *std::declval<I&>()))>>
      : std::true_type {};
#endif


#ifdef __cpp_lib_ranges
    template<typename Cont, std::ranges::input_range Rg, typename...Args> requires (not std::ranges::view<Cont>)
#else
    template<typename Cont, typename Rg, typename...Args, std::enable_if_t<
      stdex::ranges::input_range<Rg> and not stdex::ranges::view<Cont>, int> = 0>
#endif
    constexpr Cont
    to [[nodiscard]] (Rg&& r, Args&&... args)
    {
      static_assert(not std::is_const_v<Cont> and not std::is_volatile_v<Cont>);
      static_assert(std::is_class_v<Cont>);

      if constexpr (toable<Cont, Rg>)
      {
        if constexpr (stdex::constructible_from<Cont, Rg, Args...>)
        {
          return Cont(std::forward<Rg>(r), std::forward<Args>(args)...);
        }
        else if constexpr (input_iterator<Rg> and stdex::ranges::common_range<Rg> and
          stdex::constructible_from<Cont, stdex::ranges::iterator_t<Rg>, stdex::ranges::sentinel_t<Rg>, Args...>)
        {
          return Cont(stdex::ranges::begin(r), stdex::ranges::end(r), std::forward<Args>(args)...);
        }
        else
        {
          static_assert(stdex::constructible_from<Cont, Args...>);
          Cont c(std::forward<Args>(args)...);
          if constexpr (stdex::ranges::sized_range<Rg> and reservable_container<Cont>)
            c.reserve(static_cast<stdex::ranges::range_size_t<Cont>>(stdex::ranges::size(r)));
          auto it = stdex::ranges::begin(r);
          const auto sent = stdex::ranges::end(r);
          while (it != sent)
          {
#ifdef __cpp_concepts
            if constexpr (requires { c.emplace_back(*it); }) c.emplace_back(*it);
            else if constexpr (requires { c.push_back(*it); }) c.push_back(*it);
            else if constexpr (requires { c.emplace(c.end(), *it); }) c.emplace(c.end(), *it);
#else
            if constexpr (can_emplace_back<decltype(c), decltype(it)>::value) c.emplace_back(*it);
            else if constexpr (can_push_back<decltype(c), decltype(it)>::value) c.push_back(*it);
            else if constexpr (can_emplace<decltype(c), decltype(it)>::value) c.emplace(c.end(), *it);
#endif
            else c.insert(c.end(), *it);
            ++it;
          }
          return c;
        }
      }
      else
      {
        static_assert(stdex::ranges::input_range<stdex::ranges::range_reference_t<Rg>>);
        return to<Cont>(ref_view(r) | stdex::ranges::views::transform(
        [](auto&& elem) { return to<stdex::ranges::range_value_t<Cont>>(std::forward<decltype(elem)>(elem)); }), std::forward<Args>(args)...);
      }
    }


    template<typename Rg>
    struct InputIter
    {
      using iterator_category = std::input_iterator_tag;
      using value_type = stdex::ranges::range_value_t<Rg>;
      using difference_type = std::ptrdiff_t;
      using pointer = std::add_pointer_t<stdex::ranges::range_reference_t<Rg>>;
      using reference = stdex::ranges::range_reference_t<Rg>;
      reference operator*() const;
      pointer operator->() const;
      InputIter& operator++();
      InputIter operator++(int);
      bool operator==(const InputIter&) const;
    };

    template<template<typename...> typename Cont, typename Rg, typename... Args>
    using DeduceExpr1 = decltype(Cont(std::declval<Rg>(), std::declval<Args>()...));

#ifdef __cpp_concepts
    template<template<typename...> typename Cont, typename Rg, typename... Args>
    concept can_DeduceExpr1 = requires requires { typename DeduceExpr1<Cont, Rg, Args...>; };
#else
    template<template<typename...> typename Cont, typename Rg, typename = void, typename... Args>
    struct can_DeduceExpr1_impl : std::false_type {};

    template<template<typename...> typename Cont, typename Rg, typename... Args>
    struct can_DeduceExpr1_impl<Cont, Rg, std::void_t<DeduceExpr1<Cont, Rg, Args...>>, Args...> : std::true_type {};

    template<template<typename...> typename Cont, typename Rg, typename... Args>
    inline constexpr bool can_DeduceExpr1 = can_DeduceExpr1_impl<Cont, Rg, void, Args...>::value;
#endif


    template<template<typename...> typename Cont, typename Rg, typename... Args>
    using DeduceExpr3 = decltype(Cont(std::declval<InputIter<Rg>>(), std::declval<InputIter<Rg>>(), std::declval<Args>()...));

#ifdef __cpp_concepts
    template<template<typename...> typename Cont, typename Rg, typename... Args>
    concept can_DeduceExpr3 = requires requires { typename DeduceExpr3<Cont, Rg, Args...>; };
#else
    template<template<typename...> typename Cont, typename Rg, typename = void, typename... Args>
    struct can_DeduceExpr3_impl : std::false_type {};

    template<template<typename...> typename Cont, typename Rg, typename... Args>
    struct can_DeduceExpr3_impl<Cont, Rg, std::void_t<DeduceExpr3<Cont, Rg, Args...>>, Args...> : std::true_type {};

    template<template<typename...> typename Cont, typename Rg, typename... Args>
    inline constexpr bool can_DeduceExpr3 = can_DeduceExpr3_impl<Cont, Rg, void, Args...>::value;
#endif


#ifdef __cpp_concepts
    template<template<typename...> typename Cont, std::ranges::input_range Rg, typename...Args>
#else
    template<template<typename...> typename Cont, typename Rg, typename...Args, std::enable_if_t<stdex::ranges::input_range<Rg>, int> = 0>
#endif
    constexpr auto
    to [[nodiscard]] (Rg&& r, Args&&... args)
    {
      if constexpr (can_DeduceExpr1<Cont, Rg, Args...>)
        return to<DeduceExpr1<Cont, Rg, Args...>>(std::forward<Rg>(r), std::forward<Args>(args)...);
      else if constexpr (can_DeduceExpr3<Cont, Rg, Args...>)
        return to<DeduceExpr3<Cont, Rg, Args...>>(std::forward<Rg>(r), std::forward<Args>(args)...);
      else
        static_assert(false, "Cannot deduce container specialization");
    }


    template<typename Cont>
    struct To
    {
#ifdef __cpp_concepts
      template<typename Range, typename...Args> requires requires { to<Cont>(std::declval<Range&&>(), std::declval<Args&&>()...); }
#else
      template<typename Range, typename...Args, typename = std::void_t<decltype(to<Cont>(std::declval<Range&&>(), std::declval<Args&&>()...))>>
#endif
      constexpr auto
      operator()(Range&& r, Args&&... args) const
      {
        return to<Cont>(std::forward<Range>(r), std::forward<Args>(args)...);
      }
    };


#ifdef __cpp_concepts
    template<typename Cont, typename... Args> requires (not stdex::ranges::view<Cont>)
#else
    template<typename Cont, typename...Args, std::enable_if_t<not stdex::ranges::view<Cont>, int> = 0>
#endif
    constexpr auto
    to [[nodiscard]] (Args&&... args)
    {
      return Partial<To<Cont>, decay_t<Args>...>{0, std::forward<Args>(args)...};
    }


    template<template<typename...> typename Cont>
    struct To2
    {
#ifdef __cpp_concepts
      template<typename Range, typename...Args> requires requires { to<Cont>(std::declval<Range&&>(), std::declval<Args&&>()...); }
#else
      template<typename Range, typename...Args, typename = std::void_t<decltype(to<Cont>(std::declval<Range&&>(), std::declval<Args&&>()...))>>
#endif
      constexpr auto
      operator()(Range&& r, Args&&...args) const
      {
        return to<Cont>(std::forward<Range>(r), std::forward<Args>(args)...);
      }
    };


    template<template<typename...> typename Cont, typename...Args>
    constexpr auto
    to [[nodiscard]] (Args&&...args)
    {
      return Partial<To2<Cont>, decay_t<Args>...>{0, std::forward<Args>(args)...};
    }
  }


  using detail_to::to;

#endif
}


#endif
