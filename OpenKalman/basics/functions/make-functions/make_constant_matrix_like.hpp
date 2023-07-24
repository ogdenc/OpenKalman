/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_constant_matrix_like.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_MATRIX_LIKE_HPP
#define OPENKALMAN_MAKE_CONSTANT_MATRIX_LIKE_HPP

namespace OpenKalman
{
  // --------------------------- //
  //  make_constant_matrix_like  //
  // --------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Trait, typename C, typename = void, typename...D>
    struct make_constant_matrix_trait_defined: std::false_type {};

    template<typename Trait, typename C, typename...D>
    struct make_constant_matrix_trait_defined<Trait, C, std::void_t<
      decltype(Trait::template make_constant_matrix(std::declval<C&&>(), std::declval<D&&>()...))>, D...> : std::true_type {};
  }
#endif


  /**
   * \brief Make a compile-time constant matrix based on a particular library object
   * \tparam T An \indexible object (matrix or tensor) from a particular library.
   * \tparam C A \ref scalar_constant
   * \tparam Ds A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   * If no index descriptors are provided, they will be derived from T if T has no dynamic dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_constant C, index_descriptor...Ds> requires
    (sizeof...(Ds) == max_indices_of_v<T>) or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, typename C, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_constant<C> and (index_descriptor<Ds> and ...) and
    ((sizeof...(Ds) == max_indices_of_v<T>) or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>)), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(C&& c, Ds&&...ds)
  {
    if constexpr (sizeof...(Ds) == max_indices_of_v<T>)
    {
      using Trait = interface::LibraryRoutines<std::decay_t<T>>;
# ifdef __cpp_concepts
      if constexpr (requires (Ds&&...d) { Trait::template make_constant_matrix(std::forward<C>(c), std::forward<Ds>(d)...); })
# else
      if constexpr (detail::make_constant_matrix_trait_defined<Trait, C, void, Ds...>::value)
# endif
      {
        return Trait::template make_constant_matrix(std::forward<C>(c), std::forward<Ds>(ds)...);
      }
      else
      {
        // Default behavior if interface function not defined:
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        using Trait = interface::LibraryRoutines<std::decay_t<T>>;
        using U = std::decay_t<decltype(Trait::template make_default<Scalar>(std::declval<Ds&&>()...))>;
        return ConstantAdapter<U, std::decay_t<C>> {std::forward<C>(c), std::forward<Ds>(ds)...};
      }
    }
    else
    {
      return std::apply([](auto&&...arg){ return make_constant_matrix_like<T>(std::forward<decltype(arg)>(arg)...); },
        std::tuple_cat(std::forward_as_tuple(std::forward<C>(c)), get_all_dimensions_of<T>()));
    }
  }


  /**
   * \overload
   * \brief Same as above, except that the constant is derived from T, a constant object known at compile time
   */
#ifdef __cpp_concepts
  template<constant_matrix<CompileTimeStatus::known> T, index_descriptor...Ds> requires (sizeof...(Ds) == max_indices_of_v<T>)
  constexpr constant_matrix<CompileTimeStatus::known> auto
#else
  template<typename T, typename...Ds, std::enable_if_t<
    constant_matrix<T, CompileTimeStatus::known> and (index_descriptor<Ds> and ...) and
    sizeof...(Ds) == max_indices_of<T>::value, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(Ds&&...ds)
  {
    return make_constant_matrix_like<T>(constant_coefficient<T>{}, std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Make a new constant object based on a library object.
   * \tparam T The matrix or array on which the new matrix is patterned. This need not itself be constant, as only
   * its dimensions are used.
   * \tparam C A \ref scalar_constant.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_constant C>
  constexpr constant_matrix auto
#else
  template<typename T, typename C, std::enable_if_t<indexible<T> and scalar_constant<C>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t, C&& c)
  {
    return std::apply([](auto&&...arg){ return make_constant_matrix_like<T>(std::forward<decltype(arg)>(arg)...); },
      std::tuple_cat(std::forward_as_tuple(std::forward<C>(c)), get_all_dimensions_of(t)));
  }


  /**
   * \overload
   * \brief Make a compile-time constant matrix based on a particular library object and a scalar constant known at compile time
   * \tparam T A matrix or tensor from a particular library.
   * \tparam C A \ref scalar_constant for the new zero matrix. Must be constructible from {constant...}
   * \tparam constant A constant or set of coefficients in a vector space defining a constant
   * (e.g., real and imaginary parts of a complex number).
   * \param Ds A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_constant C, auto...constant, index_descriptor...Ds> requires
    ((scalar_constant<C, CompileTimeStatus::known> and sizeof...(constant) == 0) or requires { C {constant...}; }) and
    (sizeof...(Ds) == max_indices_of_v<T> or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>))
  constexpr constant_matrix auto
#else
  template<typename T, typename C, auto...constant, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_constant<C> and (index_descriptor<Ds> and ...) and
    ((scalar_constant<C, CompileTimeStatus::known> and sizeof...(constant) == 0) or std::is_constructible<C, decltype(constant)...>::value) and
    (sizeof...(Ds) == max_indices_of_v<T> or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>)), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(Ds&&...ds)
  {
    using Scalar = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;
    if constexpr (sizeof...(constant) == 0)
      return make_constant_matrix_like<T>(C{}, std::forward<Ds>(ds)...);
    else
      return make_constant_matrix_like<T>(internal::ScalarConstant<Likelihood::definitely, Scalar, constant...>{}, std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Same as above, except that the scalar type is derived from the constant template parameter
   * \tparam constant The constant
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, index_descriptor...Ds> requires scalar_type<decltype(constant)> and
    (sizeof...(Ds) == max_indices_of_v<T> or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>))
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_type<decltype(constant)> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<T> or (sizeof...(Ds) == 0 and not has_dynamic_dimensions<T>)), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(Ds&&...ds)
  {
    return make_constant_matrix_like<T, decltype(constant), constant>(std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Construct a constant object, where the shape of the new object is derived from t.
   */
#ifdef __cpp_concepts
  template<scalar_constant C, auto...constant, indexible T> requires
    ((scalar_constant<C, CompileTimeStatus::known> and sizeof...(constant) == 0) or requires { C {constant...}; })
  constexpr constant_matrix<CompileTimeStatus::known> auto
#else
  template<typename C, auto...constant, typename T, std::enable_if_t<scalar_constant<C> and indexible<T> and
    ((scalar_constant<C, CompileTimeStatus::known> and sizeof...(constant) == 0) or std::is_constructible<C, decltype(constant)...>::value), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t)
  {
    return std::apply(
      [](auto&&...arg){ return make_constant_matrix_like<T, C, constant...>(std::forward<decltype(arg)>(arg)...); },
      get_all_dimensions_of(t));
  }


/**
 * \overload
 * \brief Same as above, except that the scalar type is derived from the constant template parameter
 */
#ifdef __cpp_concepts
  template<auto constant, indexible T> requires scalar_type<decltype(constant)>
  constexpr constant_matrix<CompileTimeStatus::known> auto
#else
  template<auto constant, typename T, std::enable_if_t<scalar_type<decltype(constant)> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t)
  {
    return make_constant_matrix_like<decltype(constant), constant>(t);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_CONSTANT_MATRIX_LIKE_HPP
