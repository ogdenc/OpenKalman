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
 * \brief Definitions for \ref make_constant.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_HPP
#define OPENKALMAN_MAKE_CONSTANT_HPP

namespace OpenKalman
{
  // --------------- //
  //  make_constant  //
  // --------------- //

  /**
   * \brief Make a constant object based on a particular library object
   * \details A constant object is a matrix or tensor in which every component is the same scalar value.
   * \tparam T An \indexible object (matrix or tensor) from a particular library.
   * \tparam C A \ref scalar_constant
   * \tparam Ds A set of \ref vector_space_descriptor defining the dimensions of each index.
   * If no \ref vector_space_descriptor are provided, they will be derived from T if T has no dynamic dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_constant C, vector_space_descriptor...Ds> requires
    (sizeof...(Ds) != 0) or (not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, typename C, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_constant<C> and (vector_space_descriptor<Ds> and ...) and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant(C&& c, Ds&&...ds)
  {
    if constexpr (sizeof...(Ds) == 0)
    {
      return std::apply([](auto&&...ads){ return make_constant<T>(std::forward<decltype(ads)>(ads)...); },
        std::tuple_cat(std::forward_as_tuple(std::forward<C>(c)), all_vector_space_descriptors<T>()));
    }
    else
    {
      if constexpr (interface::make_constant_matrix_defined_for<std::decay_t<T>, C&&, Ds&&...>)
      {
        using Trait = interface::library_interface<std::decay_t<T>>;
        return Trait::template make_constant(std::forward<C>(c), std::forward<Ds>(ds)...);
      }
      else
      {
        // Default behavior if interface function not defined:
        using Scalar = std::decay_t<decltype(get_scalar_constant_value(c))>;
        auto new_dims = internal::remove_trailing_1D_descriptors(std::forward_as_tuple(std::forward<Ds>(ds)...));
        return std::apply([](C&& c, auto&&...ads){
            using U = std::decay_t<decltype(make_dense_object<T, Layout::none, Scalar>(std::declval<decltype(ads)>()...))>;
            return ConstantAdapter<U, std::decay_t<C>> {std::forward<C>(c), std::forward<decltype(ads)>(ads)...};
          }, std::tuple_cat(std::forward_as_tuple(std::forward<C>(c)), std::move(new_dims)));
      }
    }
  }


  /**
   * \overload
   * \brief Same as above, except that the constant is derived from T, a constant object known at compile time
   */
#ifdef __cpp_concepts
  template<constant_matrix<ConstantType::static_constant> T, vector_space_descriptor...Ds> requires
    (sizeof...(Ds) != 0) or (not has_dynamic_dimensions<T>)
  constexpr constant_matrix<ConstantType::static_constant> auto
#else
  template<typename T, typename...Ds, std::enable_if_t<
    constant_matrix<T, ConstantType::static_constant> and (vector_space_descriptor<Ds> and ...) and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&&...ds)
  {
    return make_constant<T>(constant_coefficient<T>{}, std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Make a new constant object based on a library object.
   * \tparam T The object on which the new matrix is patterned. This need not itself be constant, as only
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
  make_constant(const T& t, C&& c)
  {
    return std::apply([](auto&&...ads){ return make_constant<T>(std::forward<decltype(ads)>(ads)...); },
      std::tuple_cat(std::forward_as_tuple(std::forward<C>(c)), all_vector_space_descriptors(t)));
  }


  /**
   * \overload
   * \brief Make a compile-time constant based on a particular library object and a scalar constant value known at compile time
   * \tparam T A matrix or tensor from a particular library.
   * \tparam C A \ref scalar_constant for the new zero matrix. Must be constructible from {constant...}
   * \tparam constant A constant or set of coefficients in a vector space defining a constant
   * (e.g., real and imaginary parts of a complex number).
   * \param Ds A set of \ref vector_space_descriptor defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_constant C, auto...constant, vector_space_descriptor...Ds> requires
    ((scalar_constant<C, ConstantType::static_constant> and sizeof...(constant) == 0) or requires { C {constant...}; }) and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, typename C, auto...constant, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_constant<C> and (vector_space_descriptor<Ds> and ...) and
    ((scalar_constant<C, ConstantType::static_constant> and sizeof...(constant) == 0) or std::is_constructible<C, decltype(constant)...>::value) and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&&...ds)
  {
    using Scalar = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;
    if constexpr (sizeof...(constant) == 0)
      return make_constant<T>(C{}, std::forward<Ds>(ds)...);
    else
      return make_constant<T>(internal::ScalarConstant<Qualification::unqualified, Scalar, constant...>{}, std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Same as above, except that the scalar type is derived from the constant template parameter
   * \tparam constant The constant
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, vector_space_descriptor...Ds> requires scalar_type<decltype(constant)> and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_type<decltype(constant)> and (vector_space_descriptor<Ds> and ...) and
    (sizeof...(Ds) != 0 or not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant(Ds&&...ds)
  {
    return make_constant<T, decltype(constant), constant>(std::forward<Ds>(ds)...);
  }


  /**
   * \overload
   * \brief Construct a constant object, where the shape of the new object is derived from t.
   */
#ifdef __cpp_concepts
  template<scalar_constant C, auto...constant, indexible T> requires
    ((scalar_constant<C, ConstantType::static_constant> and sizeof...(constant) == 0) or requires { C {constant...}; })
  constexpr constant_matrix<ConstantType::static_constant> auto
#else
  template<typename C, auto...constant, typename T, std::enable_if_t<scalar_constant<C> and indexible<T> and
    ((scalar_constant<C, ConstantType::static_constant> and sizeof...(constant) == 0) or std::is_constructible<C, decltype(constant)...>::value), int> = 0>
  constexpr auto
#endif
  make_constant(const T& t)
  {
    return std::apply(
      [](auto&&...arg){ return make_constant<T, C, constant...>(std::forward<decltype(arg)>(arg)...); },
      all_vector_space_descriptors(t));
  }


/**
 * \overload
 * \brief Same as above, except that the scalar type is derived from the constant template parameter
 */
#ifdef __cpp_concepts
  template<auto constant, indexible T> requires scalar_type<decltype(constant)>
  constexpr constant_matrix<ConstantType::static_constant> auto
#else
  template<auto constant, typename T, std::enable_if_t<scalar_type<decltype(constant)> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_constant(const T& t)
  {
    return make_constant<decltype(constant), constant>(t);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_CONSTANT_HPP
