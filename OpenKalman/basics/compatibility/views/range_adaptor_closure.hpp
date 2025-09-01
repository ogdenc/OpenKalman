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
 * \brief Definition of \ref stdcompat::ranges::range_adaptor_closure.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_RANGE_ADAPTOR_CLOSURE_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_RANGE_ADAPTOR_CLOSURE_HPP

#include <type_traits>
#include "basics/compatibility/language-features.hpp"
#include "view-concepts.hpp"

namespace OpenKalman::stdcompat::ranges
{
#if __cpp_lib_ranges >= 202202L
  using std::ranges::range_adaptor_closure;
#else
#ifdef __cpp_concepts
  template<typename D> requires std::is_object_v<D> && std::same_as<D, std::remove_cv_t<D>>
#else
  template<typename D>
#endif
  struct range_adaptor_closure {};


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U> requires (not std::same_as<T, range_adaptor_closure<U>>)
#else
    template<typename T, typename U, std::enable_if_t<not std::is_same_v<T, range_adaptor_closure<U>>, int> = 0>
#endif
    void range_adaptor_closure_fn (const T&, const range_adaptor_closure<U>&);


#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_range_adaptor_closure_impl : std::false_type {};

    template<typename T>
    struct is_range_adaptor_closure_impl<T,
      std::void_t<decltype(detail::range_adaptor_closure_fn(std::declval<T>(), std::declval<T>()))>> : std::true_type {};
#endif


    template<typename T>
#ifdef __cpp_concepts
    concept is_range_adaptor_closure = requires (T t) { detail::range_adaptor_closure_fn(t, t); };
#else
    inline constexpr bool is_range_adaptor_closure = detail::is_range_adaptor_closure_impl<T>::value;
#endif


#ifndef __cpp_concepts
    template<typename Lhs, typename Rhs, typename R, typename = void>
    struct is_pipe_invocable : std::false_type {};

    template<typename Lhs, typename Rhs, typename R>
    struct is_pipe_invocable<Lhs, Rhs, R, std::void_t<decltype(std::declval<Rhs>()(std::declval<Lhs>()(std::declval<R>())))>> : std::true_type {};
#endif


    template<typename Lhs, typename Rhs, typename R>
#ifdef __cpp_concepts
    concept __pipe_invocable = requires { std::declval<Rhs>()(std::declval<Lhs>()(std::declval<R>())); };
#else
    inline constexpr bool pipe_invocable = is_pipe_invocable<Lhs, Rhs, R>::value;
#endif

  }


  namespace internal
  {
    template<typename Lhs, typename Rhs>
    struct Pipe : range_adaptor_closure<Pipe<Lhs, Rhs>>
    {
      template<typename T, typename U>
    	constexpr
	    Pipe(T&& lhs, U&& rhs) : my_lhs(std::forward<T>(lhs)), my_rhs(std::forward<U>(rhs)) {}


#if __cpp_explicit_this_parameter
      template<typename Self, typename R>
	    requires pipe_invocable<decltype(std::forward_like<Self>(std::declval<Lhs>())), decltype(std::forward_like<Self>(std::declval<Rhs>())), R>
	    constexpr auto
	    operator()(this Self&& self, R&& r)
	    {
	      return (std::forward<Self>(self).my_rhs(std::forward<Self>(self).my_lhs(std::forward<R>(r))));
	    }
#else
      template<typename R, std::enable_if_t<detail::pipe_invocable<const Lhs&, const Rhs&, R>, int> = 0>
	    constexpr auto
	    operator()(R&& r) const & { return my_rhs(my_lhs(std::forward<R>(r))); }


      template<typename R, std::enable_if_t<detail::pipe_invocable<Lhs, Rhs, R>, int> = 0>
	    constexpr auto
	    operator()(R&& r) && { return std::move(my_rhs)(std::move(my_lhs)(std::forward<R>(r))); }


      template<typename R>
	    constexpr auto
	    operator()(R&& r) const && = delete;

    private:

      [[no_unique_address]] Lhs my_lhs;
      [[no_unique_address]] Rhs my_rhs;
#endif
    };

  }


#ifdef __cpp_concepts
  template<detail::is_range_adaptor_closure S, typename R> requires requires { std::declval<S>()(declval<R>()); }
#else
  template<typename S, typename R, std::enable_if_t<detail::is_range_adaptor_closure<S> and not detail::is_range_adaptor_closure<R>, int> = 0>
#endif
  constexpr auto
  operator | (R&& r, S&& s)
  {
    return std::forward<S>(s)(std::forward<R>(r));
  }


#ifdef __cpp_concepts
  template<detail::is_range_adaptor_closure Lhs, detail::is_range_adaptor_closure Rhs>
#else
  template<typename Lhs, typename Rhs, std::enable_if_t<detail::is_range_adaptor_closure<Lhs> and detail::is_range_adaptor_closure<Rhs>, int> = 0>
#endif
  constexpr auto
  operator | (Lhs&& lhs, Rhs&& rhs)
  {
    return internal::Pipe<std::decay_t<Lhs>, std::decay_t<Rhs>>{std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)};
  }

#endif
}

#endif
