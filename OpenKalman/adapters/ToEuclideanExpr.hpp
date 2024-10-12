/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief ToEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_TOEUCLIDEANEXPR_HPP
#define OPENKALMAN_TOEUCLIDEANEXPR_HPP

namespace OpenKalman
{

  /// \todo Remove nested diagonal matrix option
#ifdef __cpp_concepts
  template<indexible NestedObject> requires (not from_euclidean_expr<NestedMatrix>) 
#else
  template<typename NestedObject>
#endif
  struct ToEuclideanExpr : internal::AdapterBase<ToEuclideanExpr<NestedObject>, NestedObject>
  {

  private:

#ifndef __cpp_concepts
    static_assert(not from_euclidean_expr<NestedObject>);
#endif

    using Scalar = scalar_type_of_t<NestedObject>;

    using Base = internal::AdapterBase<ToEuclideanExpr, NestedObject>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr ToEuclideanExpr() requires std::default_initializable<Base>
#else
    template<typename B = Base, std::enable_if_t<std::is_default_constructible_v<B>, int> = 0>
    constexpr ToEuclideanExpr()
#endif
    {}


    /// Construct from compatible \ref indexible object.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not to_euclidean_expr<Arg>) and std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<not to_euclidean_expr<Arg> and
      indexible<Arg> and std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit ToEuclideanExpr(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /// Assign from a compatible \ref indexible object.
#ifdef __cpp_concepts
    template<indexible Arg> requires 
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and 
      (index_dimension_of_v<Arg, 0> == euclidean_dimension_size_of_v<Descriptor>) and
      (index_dimension_of_v<Arg, 1> == columns) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(from_euclidean<Descriptor>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and 
      (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and 
      (index_dimension_of<Arg, 0>::value == euclidean_dimension_size_of_v<Descriptor>) and (index_dimension_of<Arg, 1>::value == columns) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(from_euclidean<Descriptor>(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero<NestedObject> and not identity_matrix<NestedObject>)
      {
        this->nested_object() = from_euclidean<Descriptor>(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Increment from another \ref to_euclidean_expr.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      this->nested_object() = from_euclidean<Descriptor>(*this + arg);
      return *this;
    }


    /// Increment from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not to_euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == euclidean_dimension_size_of_v<Descriptor>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not to_euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == euclidean_dimension_size_of_v<Descriptor>), int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      this->nested_object() = from_euclidean<Descriptor>(*this + arg);
      return *this;
    }


    /// Decrement from another \ref to_euclidean_expr.
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires (index_dimension_of_v<Arg, 1> == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and (index_dimension_of<Arg, 1>::value == columns) and
      equivalent_to<vector_space_descriptor_of_t<Arg, 0>, Descriptor>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      this->nested_object() = from_euclidean<Descriptor>(*this - arg);
      return *this;
    }


    /// Decrement from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not to_euclidean_expr<Arg>) and (index_dimension_of_v<Arg, 1> == columns) and
      (index_dimension_of_v<Arg, 0> == euclidean_dimension_size_of_v<Descriptor>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not to_euclidean_expr<Arg>) and
      (index_dimension_of<Arg, 1>::value == columns) and
      (index_dimension_of<Arg, 0>::value == euclidean_dimension_size_of_v<Descriptor>), int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      this->nested_object() = from_euclidean<Descriptor>(*this - arg);
      return *this;
    }


    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      this->nested_object() = from_euclidean<Descriptor>(scalar_product(*this, scale));
      return *this;
    }


    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_object() = from_euclidean<Descriptor>(scalar_quotient(*this, scale));
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<indexible Arg, vector_space_descriptor C>
#else
  template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
  ToEuclideanExpr(Arg&&, const C&) -> ToEuclideanExpr<C, passable_t<Arg>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Descriptor, typename NestedObject>
    struct indexible_object_traits<ToEuclideanExpr<Descriptor, NestedObject>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (n == 0_uz) return std::forward<Arg>(arg).my_dimension;
          else return OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Scalar = scalar_type_of<Arg>;
          if (n == 0) return DynamicDescriptor<Scalar> {std::forward<Arg>(arg).my_dimension};
          else return DynamicDescriptor<Scalar> {OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n)};
        }
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedObject>)
          return constant_coefficient{arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_vector_space_descriptor<NestedObject>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<Qualification b>
      static constexpr bool one_dimensional = euclidean_vector_space_descriptor<Descriptor> and OpenKalman::one_dimensional<NestedObject, b>;


      template<TriangleType t>
      static constexpr bool is_triangular = euclidean_vector_space_descriptor<Descriptor> and triangular_matrix<NestedObject, t>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = hermitian_matrix<NestedObject> and euclidean_vector_space_descriptor<Descriptor>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_object_of_t<Arg&&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_gettable<typename nested_object_of<Arg&&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr auto get(Arg&& arg, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
        {
          return get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), i, is...);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), ix, is...); }};
          return to_euclidean_element(get_vector_space_descriptor<0>(arg), g, i, 0);
        }
      }


      /**
       * \internal
       * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
       * \details This function sets the nested matrix, not the wrapped resulting matrix.
       * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
       * single-column vector is {-1., pi/2}, then set_component(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
       * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
       * \tparam Arg The matrix to set.
       * \tparam Scalar The value to set the coefficient to.
       * \param i The row of the coefficient.
       * \param j The column of the coefficient.
       */
  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_gettable<nested_object_of_t<Arg&>, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_object_of_t<Arg>>))
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<
        element_gettable<typename nested_object_of<Arg&>::type, 1 + sizeof...(Is)> and
        (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_object_of_t<Arg>>)), int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        if constexpr (has_untyped_index<Arg, 0>)
          set_component(OpenKalman::nested_object(OpenKalman::nested_object(arg)), s, i, is...);
        else
          set_component(OpenKalman::nested_object(arg), s, i, is...);
      }


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires has_untyped_index<Arg, 0> and raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<has_untyped_index<Arg, 0> and raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(OpenKalman::nested_object(arg)); }


      static constexpr Layout layout = euclidean_vector_space_descriptor<Descriptor> ? layout_of_v<NestedObject> : Layout::none;

    };

  } // namespace interface


} // OpenKalman



#endif //OPENKALMAN_TOEUCLIDEANEXPR_HPP
