/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to a compatible replacement for std::invoke.
 */

#ifndef OPENKALMAN_COMPATIBILITY_INVOKE_HPP
#define OPENKALMAN_COMPATIBILITY_INVOKE_HPP

namespace OpenKalman::stdex
{
#if __cplusplus >= 202002L
  using std::invoke;
#else
  namespace detail
  {
    template<typename> static constexpr bool is_reference_wrapper_v = false;
    template<typename U> static constexpr bool is_reference_wrapper_v<std::reference_wrapper<U>> = true;
    template<typename U> static constexpr bool is_reference_wrapper_v<stdex::reference_wrapper<U>> = true;

    template<typename C, typename Pointed, typename Object, typename...Args>
    static constexpr decltype(auto)
    invoke_memptr(Pointed C::* member, Object&& object, Args&&... args)
    {
      using object_t = remove_cvref_t<Object>;
      constexpr bool is_member_function = std::is_function_v<Pointed>;
      constexpr bool is_wrapped = is_reference_wrapper_v<object_t>;
      constexpr bool is_derived_object = std::is_same_v<C, object_t> or std::is_base_of_v<C, object_t>;

      if constexpr (is_member_function)
      {
        if constexpr (is_derived_object)
          return (std::forward<Object>(object) .* member) (std::forward<Args>(args)...);
        else if constexpr (is_wrapped)
          return (object.get() .* member)(std::forward<Args>(args)...);
        else
          return ((*std::forward<Object>(object)) .* member) (std::forward<Args>(args)...);
      }
      else
      {
        static_assert(std::is_object_v<Pointed> && sizeof...(args) == 0);
        if constexpr (is_derived_object)
          return std::forward<Object>(object) .* member;
        else if constexpr (is_wrapped)
          return object.get() .* member;
        else
          return (*std::forward<Object>(object)) .* member;
      }
    }
  }


  /**
   * \internal
   * \brief A constexpr version of std::invoke, for use when compiling in c++17
   **/
  template<typename F, typename...Args>
  static constexpr std::invoke_result_t<F, Args...>
  invoke(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_v<F, Args...>)
  {
    if constexpr (std::is_member_pointer_v<stdex::remove_cvref_t<F>>)
      return detail::invoke_memptr(f, std::forward<Args>(args)...);
    else
      return std::forward<F>(f)(std::forward<Args>(args)...);
  }
#endif


#if __cplusplus >= 202302L
  using std::invoke_r;
#else
  /**
   * \internal
   * \brief A constexpr version of std::invoke_r, for use when compiling in c++17
   **/
#ifdef __cpp_concepts
  template<typename R, typename F, typename...Args> requires std::is_invocable_r_v<R, F, Args...>
#else
  template<typename R, typename F, typename...Args, std::enable_if_t<std::is_invocable_r_v<R, F, Args...>, int> = 0>
#endif
  constexpr R invoke_r(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_r_v<R, F, Args...>)
  {
    if constexpr (std::is_void_v<R>)
      stdex::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    else
      return stdex::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
#endif

}


#endif