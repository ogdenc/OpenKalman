/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief HermitianAdapter and related definitions.
 */

#ifndef OPENKALMAN_HERMITIANADAPTER_HPP
#define OPENKALMAN_HERMITIANADAPTER_HPP

#include "basics/basics.hpp"

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_shaped<Applicability::permitted> NestedObject, HermitianAdapterType storage_triangle> requires
    (index_count_v<NestedObject> <= 2) and
    (storage_triangle == HermitianAdapterType::lower or storage_triangle == HermitianAdapterType::upper) and
    (not constant_matrix<NestedObject> or values::not_complex<constant_coefficient<NestedObject>>) and
    (not constant_diagonal_matrix<NestedObject> or values::not_complex<constant_diagonal_coefficient<NestedObject>>) and
    (not triangular_matrix<NestedObject, TriangleType::any> or triangular_matrix<NestedObject, static_cast<TriangleType>(storage_triangle)>)
#else
  template<typename NestedObject, HermitianAdapterType storage_triangle>
#endif
  struct HermitianAdapter
    : OpenKalman::internal::AdapterBase<HermitianAdapter<NestedObject, storage_triangle>, NestedObject>
  {

#ifndef __cpp_concepts
    static_assert(square_shaped<NestedObject, Applicability::permitted>);
    static_assert(index_count_v<NestedObject> <= 2);
    static_assert(storage_triangle == HermitianAdapterType::lower or storage_triangle == HermitianAdapterType::upper);
    static_assert([]{if constexpr (constant_matrix<NestedObject>) return values::not_complex<constant_coefficient<NestedObject>>; else return true; }());
    static_assert([]{if constexpr (constant_diagonal_matrix<NestedObject>) return values::not_complex<constant_diagonal_coefficient<NestedObject>>; else return true; }());
    static_assert(not triangular_matrix<NestedObject, TriangleType::any> or triangular_matrix<NestedObject, static_cast<TriangleType>(storage_triangle)>);
#endif


  private:

    using Base = OpenKalman::internal::AdapterBase<HermitianAdapter, NestedObject>;

    static constexpr auto dim = dynamic_dimension<NestedObject, 0> ? index_dimension_of_v<NestedObject, 1> :
      index_dimension_of_v<NestedObject, 0>;


  public:

    using Scalar = scalar_type_of_t<NestedObject>;


    /// Default constructor.
#ifdef __cpp_concepts
    HermitianAdapter() requires std::default_initializable<NestedObject> and (not has_dynamic_dimensions<NestedObject>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      stdcompat::default_initializable<NestedObject> and (not has_dynamic_dimensions<NestedObject>), int> = 0>
    HermitianAdapter()
#endif
      : Base {} {}


    /// Construct from a diagonal matrix if NestedObject is a \ref diagonal_adapter.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      diagonal_adapter<NestedObject> and vector_space_descriptors_may_match_with<Arg, NestedObject> and
      requires(Arg&& arg) { NestedObject {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      diagonal_matrix<Arg> and diagonal_adapter<NestedObject> and vector_space_descriptors_match_with<Arg, NestedObject> and
      stdcompat::constructible_from<NestedObject, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a diagonal matrix if NestedObject is not diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      (not diagonal_matrix<NestedObject> or
        not requires(Arg&& arg) { NestedObject {diagonal_of(std::forward<Arg>(arg))}; }) and
      std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      (not diagonal_matrix<NestedObject> or
        not stdcompat::constructible_from<NestedObject, decltype(diagonal_of(std::declval<Arg&&>()))>) and
      stdcompat::constructible_from<NestedObject, Arg&&>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian, non-diagonal wrapper of the same storage type
#ifdef __cpp_concepts
    template<hermitian_adapter<storage_triangle> Arg> requires (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      (not diagonal_matrix<Arg>) and square_shaped<nested_object_of_t<Arg>, Applicability::permitted> and
      std::constructible_from<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedObject {nested_object(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<
      hermitian_adapter<Arg, storage_triangle> and (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      (not diagonal_matrix<Arg>) and square_shaped<nested_object_of_t<Arg>, Applicability::permitted> and
      stdcompat::constructible_from<NestedObject, decltype(nested_object(std::declval<Arg&&>()))>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a hermitian, non-diagonal wrapper of the opposite storage type
#ifdef __cpp_concepts
    template<hermitian_adapter Arg> requires (not diagonal_matrix<Arg>) and
      (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_shaped<nested_object_of_t<Arg>, Applicability::permitted> and
      requires(Arg&& arg) { NestedObject {transpose(nested_object(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<
      hermitian_adapter<Arg> and (not diagonal_matrix<Arg>) and
      (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_shaped<nested_object_of_t<Arg>, Applicability::permitted> and
      stdcompat::constructible_from<NestedObject, decltype(transpose(nested_object(std::declval<Arg&&>())))>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {transpose(nested_object(std::forward<Arg>(arg)))} {}


    /// Construct from a hermitian matrix of the same storage type and is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_object<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_object<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      stdcompat::constructible_from<NestedObject, Arg&&>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian matrix, of the opposite storage type, that is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_object<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      requires(Arg&& arg) { NestedObject {transpose(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_object<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      stdcompat::constructible_from<NestedObject, decltype(transpose(std::declval<Arg&&>()))>, int> = 0>
#endif
    HermitianAdapter(Arg&& arg) : Base {transpose(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedObject is not diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Applicability::permitted> Arg> requires (not hermitian_adapter<Arg>) and (not diagonal_matrix<NestedObject>) and
      std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Applicability::permitted> and
      not hermitian_adapter<Arg> and not diagonal_matrix<NestedObject> and
      stdcompat::constructible_from<NestedObject, Arg&&>, int> = 0>
#endif
    explicit HermitianAdapter(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_dimension<Arg, 0> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_dimension<Arg, 1> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedObject is diagonal.
#ifdef __cpp_concepts
    template<square_shaped<Applicability::permitted> Arg> requires (not hermitian_matrix<Arg>) and diagonal_matrix<NestedObject> and
      requires(Arg&& arg) { NestedObject {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_shaped<Arg, Applicability::permitted> and (not hermitian_matrix<Arg>) and
      diagonal_matrix<NestedObject> and stdcompat::constructible_from<NestedObject, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit HermitianAdapter(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_dimension<Arg, 0> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_dimension<Arg, 1> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedObject {make_dense_object_from<NestedObject>(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Args, const Scalar>...> and (sizeof...(Args) > 0) and
      (stdcompat::constructible_from<NestedObject,
        dense_writable_matrix_t<NestedObject, Layout::none, Scalar,
          std::tuple<
            Dimensions<static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>,
            Dimensions<static_cast<std::size_t>(values::sqrt(sizeof...(Args)))>>>> or
        (diagonal_matrix<NestedObject> and stdcompat::constructible_from<NestedObject,
          dense_writable_matrix_t<NestedObject, Layout::none, Scalar, std::tuple<Dimensions<sizeof...(Args)>, Axis>>>)), int> = 0>
#endif
    HermitianAdapter(Args ... args)
      : Base {make_dense_object_from<NestedObject>(static_cast<const Scalar>(args)...)} {}


    /// Assign from another \ref hermitian_matrix.
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and
      vector_space_descriptors_may_match_with<NestedObject, Arg> and (not diagonal_matrix<NestedObject> or diagonal_matrix<Arg>) and
      (writable<NestedObject> or std::assignable_from<NestedObject, Arg&&>)
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and
      (not std::is_base_of_v<HermitianAdapter, std::decay_t<Arg>>) and vector_space_descriptors_may_match_with<NestedObject, Arg> and
      (not diagonal_matrix<NestedObject> or diagonal_matrix<Arg>) and
      (writable<NestedObject> or std::is_assignable_v<NestedObject, Arg&&>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (writable<NestedObject>) internal::set_triangle<storage_triangle>(this->nested_object(), std::forward<Arg>(arg));
      else if (hermitian_adapter_type_of_v<Arg> != storage_triangle) Base::operator=(adjoint(std::forward<Arg>(arg)));
      else Base::operator=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<NestedObject> Arg, HermitianAdapterType t> requires diagonal_matrix<Arg> or
      (not diagonal_matrix<NestedObject>)
#else
    template<typename Arg, HermitianAdapterType t, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject> and
      (diagonal_matrix<Arg> or (not diagonal_matrix<NestedObject>)), int> = 0>
#endif
    auto& operator+=(const HermitianAdapter<Arg, t>& arg)
    {
      if constexpr (writable<NestedObject>) internal::set_triangle<storage_triangle>(this->nested_object(), this->nested_object() + std::forward<Arg>(arg));
      else if (t != storage_triangle) Base::operator+=(adjoint(std::forward<Arg>(arg)));
      else Base::operator+=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<NestedObject> Arg, HermitianAdapterType t> requires diagonal_matrix<Arg> or
      (not diagonal_matrix<NestedObject>)
#else
    template<typename Arg, HermitianAdapterType t, std::enable_if_t<vector_space_descriptors_may_match_with<Arg, NestedObject> and
      (diagonal_matrix<Arg> or (not diagonal_matrix<NestedObject>)), int> = 0>
#endif
    auto& operator-=(const HermitianAdapter<Arg, t>& arg)
    {
      if constexpr (writable<NestedObject>) internal::set_triangle<storage_triangle>(this->nested_object(), this->nested_object() - std::forward<Arg>(arg));
      else if (t != storage_triangle) Base::operator-=(adjoint(std::forward<Arg>(arg)));
      else Base::operator-=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<stdcompat::convertible_to<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr (writable<NestedObject>) internal::set_triangle<storage_triangle>(this->nested_object(), scalar_product(this->nested_object(), s));
      else Base::operator*=(s);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<stdcompat::convertible_to<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr (writable<NestedObject>) internal::set_triangle<storage_triangle>(this->nested_object(), scalar_quotient(this->nested_object(), s));
      else Base::operator/=(s);
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, HermitianAdapter>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(-nested_object(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return make_hermitian_matrix<storage_triangle>(-nested_object(*this));
    }

    decltype(auto) operator-() const&&
    {
      return make_hermitian_matrix<storage_triangle>(-nested_object(std::move(*this)));
    }
#endif


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, HermitianAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, HermitianAdapter> and stdcompat::convertible_to<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(nested_object(std::forward<Arg>(arg)) * s);
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, HermitianAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, HermitianAdapter> and stdcompat::convertible_to<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(s * nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, HermitianAdapter>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, HermitianAdapter> and stdcompat::convertible_to<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(nested_object(std::forward<Arg>(arg)) / s);
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<hermitian_matrix<Applicability::permitted> M>
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M, Applicability::permitted>, int> = 0>
#endif
  explicit HermitianAdapter(M&&) -> HermitianAdapter<
    std::conditional_t<hermitian_adapter<M>, nested_object_of_t<M>, M>,
    hermitian_adapter<M> ? hermitian_adapter_type_of_v<M> : HermitianAdapterType::lower>;


#ifdef __cpp_concepts
  template<triangular_matrix M> requires (not hermitian_matrix<M, Applicability::permitted>)
#else
  template<typename M, std::enable_if_t<triangular_matrix<M> and
    (not hermitian_matrix<M, Applicability::permitted>), int> = 0>
#endif
  explicit HermitianAdapter(M&&) -> HermitianAdapter<
    std::conditional_t<triangular_adapter<M>, nested_object_of_t<M>, M>,
    triangular_matrix<M, TriangleType::lower> ? HermitianAdapterType::lower : HermitianAdapterType::upper>;


#ifdef __cpp_concepts
  template<indexible M> requires
    (not hermitian_matrix<M, Applicability::permitted>) and (not triangular_matrix<M>)
#else
  template<typename M, std::enable_if_t<indexible<M> and
      (not hermitian_matrix<M, Applicability::permitted>) and (not triangular_matrix<M>), int> = 0>
#endif
  explicit HermitianAdapter(M&&) -> HermitianAdapter<M, HermitianAdapterType::lower>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename NestedObject, HermitianAdapterType storage_type>
    struct indexible_object_traits<HermitianAdapter<NestedObject, storage_type>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        return internal::most_fixed_pattern(
          OpenKalman::get_vector_space_descriptor<0>(nested_object(arg)),
          OpenKalman::get_vector_space_descriptor<1>(nested_object(arg)));
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{OpenKalman::nested_object(arg)};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_diagonal_coefficient {OpenKalman::nested_object(arg)};
      }


      template<Applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedObject, b>;


      template<Applicability b>
      static constexpr bool is_square = true;


      template<TriangleType t>
      static constexpr bool is_triangular = triangular_matrix<NestedObject, TriangleType::diagonal>;


      static constexpr bool is_hermitian = true;


      static constexpr HermitianAdapterType hermitian_adapter_type = storage_type;


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(nested_object(arg)); }


      static constexpr Layout layout = OpenKalman::one_dimensional<NestedObject> ? layout_of_v<NestedObject> : Layout::none;

    };

  } // namespace interface

} // namespace OpenKalman



#endif //OPENKALMAN_HERMITIANADAPTER_HPP

